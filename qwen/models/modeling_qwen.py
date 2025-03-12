# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import importlib
import math
from typing import TYPE_CHECKING, Optional, Tuple, Union, Callable, List

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

'''
Automatic Mixed Precision（自动混合精度）的缩写，
它可以在不损失太多精度的情况下，加快模型训练速度并减少内存使用。
autocast 是 torch.cuda.amp 中的一个上下文管理器。
当你使用 autocast 时，它会自动在适当的操作上使用半精度（float16）计算，
这样可以利用 GPU 的加速能力，因为很多 GPU 对 float16 的计算速度更快
'''
from torch.cuda.amp import autocast

from torch.nn import CrossEntropyLoss
from transformers import PreTrainedTokenizer, GenerationConfig, StoppingCriteriaList
from transformers.generation.logits_process import LogitsProcessorList

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging


try:
    from einops import rearrange
except ImportError:
    rearrange = None
from torch import nn


SUPPORT_CUDA = torch.cuda.is_available()
SUPPORT_BF16 = SUPPORT_CUDA and torch.cuda.is_bf16_supported()
SUPPORT_FP16 = SUPPORT_CUDA and torch.cuda.get_device_capability(0)[0] >= 7


apply_rotary_emb_func = None
rms_norm = None
flash_attn_unpadded_func = None


from .configuration_qwen import QWenConfig
from .qwen_generation_utils import (
    HistoryType,
    make_context,
    decode_tokens,
    get_stop_words_ids,
    StopWordsLogitsProcessor,
)


logger = logging.get_logger(__name__)


_CHECKPOINT_FOR_DOC = "qwen"
_CONFIG_FOR_DOC = "QWenConfig"
QWen_PRETRAINED_MODEL_ARCHIVE_LIST = ["qwen-7b"]




class FlashSelfAttention(nn.Module):
    def __init__(
        self, 
        causal=False,
        softmax_scale=None,
        attention_dropout=0.0,
        ):
        super().__init__()
        assert flash_attn_unpadded_func is not None, (
            "Please install FlashAttention first, " "e.g., with pip install flash-attn"
        )
        assert (
            rearrange is not None
        ), "Please install einops first, e.g., with pip install einops"
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        
        
    def forward(self, q, k, v):
        assert all((i.dtype in [torch.float16, torch.bfloat16] for i in (q, k, v)))
        assert all((i.is_cuda for i in (q, k, v)))
        batch_size, seqlen_q = q.shape[0], q.shape[1]
        seqlen_k = k.shape[1]
        # rearrange 是 einops 库中的一个函数，用于对张量进行维度的重新排列和重塑。
            # b 表示 batch size（批次大小）
            # s 表示 sequence length（序列长度）
            # ... 表示剩余的维度
            # 具体来说，它将 b 和 s 这两个维度合并为一个维度 (b s)，而保持其他维度不变。  
        q, k, v = [rearrange(x, "b s ... -> (b s) ...") for x in [q, k, v]]
        
        # FlashAttention 的输入：FlashAttention 需要知道每个序列在合并后的张量中的位置，以便正确处理不同长度的序列。
        
        # 它是一个一维张量，长度为 batch_size + 1，其中每个元素表示当前批次中所有序列的累积长度
        # 例如，如果批次中有两个序列，长度分别为 3 和 5，那么 cu_seqlens_q 将是 [0, 3, 8]。
        # 它的主要作用是告诉 FlashAttention 每个序列在合并后的张量中的起始和结束位置。
        cu_seqlens_q = torch.arange( # 用于 FlashAttention 的累积序列长度（cumulative sequence lengths）数组
            0,
            (batch_size + 1) * seqlen_q,
            step=seqlen_q,
            dtype=torch.int32,
            device=q.device,
        )
        
        if self.training:
            assert seqlen_k == seqlen_q
            is_causal = self.causal
            cu_seqlens_k = cu_seqlens_q
        
        
        else:
            is_causal = seqlen_q == seqlen_k
            cu_seqlens_k = torch.arange(
                0,
                (batch_size +1)* seqlen_k,
                step = seqlen_k,
                dtype = torch.int32,
                device=q.device,
            )
            
            self.dropout_p = 0
            
        
        output = flash_attn_unpadded_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlen_q,
            seqlen_k,
            self.dropout_p,
            softmax_scale=self.softmax_scale,
            causal=is_causal,
        )
        
        output = rearrange(output, "(b s) ... -> b s ...", b=batch_size) # shape = [batch_size, seqlen_q, hidden_size]

        return output



class QWenAttention(nn.Module):
    def __init__(self, config: QWenConfig, layer_number = None):
        super().__init__()
        
        max_positions = config.max_position_embeddings
        
        self.register_buffer(
            "bias", torch.tril(
                torch.ones((max_positions, max_positions)).view(1,1,max_positions,max_positions)
            ),
            persistent=False, # 设置缓冲区为非持久化，即不会被保存到模型的状态字典中
        )

        
        self.register_buffer(
            "masked_bias", torch.tril(
                torch.tensor(1e-4)
            ),
            persistent=False,
        )
        
        self.layer_number = max(1, layer_number)
        
        self.params_dtype = config.params_dtype
        self.seq_length = config.seq_length

        self.hidden_size = config.hidden_size
        self.split_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        self.use_flash_attn = config.use_flash_attn
        self.scale_attn_weights = True
        
        self.layer_idx = None
        
        self.projection_size = config.kv_channels * config.num_attention_heads













class QWenMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(
            config.hidden_size, config.ffn_hidden_size // 2, bias=not config.no_bias
        )
        self.w2 = nn.Linear(
            config.hidden_size, config.ffn_hidden_size // 2, bias=not config.no_bias
        )
        ff_dim_in = config.ffn_hidden_size // 2
        self.c_proj = nn.Linear(ff_dim_in, config.hidden_size, bias=not config.no_bias)

        
    def forward(self, hidden_states):
        a1 = self.w1(hidden_states)
        a2 = self.w2(hidden_states)
        
        intermediate_parallel = a1 * F.silu(a2)
        output = self.c_proj(intermediate_parallel)
        
        return output
    
    
    
    





class QwenBlock(nn.Module):
    def __init__(self, config: QWenConfig, layer_idx=None, num_expert=1):
        super().__init__()
        self.layer_number = layer_idx
        self.num_expert = num_expert
        self.hidden_size = config.hidden_size
        
        self.apply_residual_connection_post_layernorm = (
            config.apply_residual_connection_post_layernorm
        )
        
        self.bf16 = config.bf16
        
        self.ln_1 = RMSNorm(
            
        )
        
        self.attn = QWenAttention(
            config = config, layer_number = layer_idx,
        )
        
        self.ln_2 = RMSNorm(
            
        )
        
        self.mlp = QWenMLP(
            config = config,
        )
        
        
    
    
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        layernorm_output = self.ln_1(hidden_states)

        attn_outputs = self.attn(
            
        )
    
    
    





class QwenPretrainedModel(PreTrainedModel):
    config_class = QWenConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _no_split_modules = ["QWenBlock"]
    
    def __init__(self, *inputs, **kwargs):
        super().__init__(inputs, kwargs)

    
    def _init_weights(self, module):
        """Initialize the weights."""
        pass
    
    
    
    
    def _set_gradient_checkpointing(self, module, value = False):
        if isinstance(module, QwenModel):
            module.graadient_checkpointing = value
        
    
    
    
class QwenModel(QwenPretrainedModel):
    def __init__(self, config:QWenConfig):
        super().__init__(config)

        self.vocab_size = config.padded_vocab_size
        self.num_hidden_layers = config.num_hidden_layers
        self.embed_dim = config.hidden_size

        max_sequence_length = config.max_position_embeddings
        self.position_embedding_type = config.pos_emb
        self.gradient_checkpointing = False
    
    
        if self.position_embedding_type == "learned":
            self.wpe = nn.Embedding(max_sequence_length, self.embed_dim)
        else:
            self.wpe = None
            
        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        
        self.dropout = nn.Dropout(config.embd_pdrop)
        
        self.h = nn.ModuleList(
            []
        )
        
        
        
        
        














class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight =  nn.Parameter(torch.ones(dim))
        
        
    def _norm(self, x):
        pass
    
    
    
    
    def forward(self, x):
        if rms_norm is not None and x.is_cuda:
            return rms_norm(x, self.weight, self.eps)
        else:
            output = self._norm(x.float()).type_as(x)
            
        return output * self.weight