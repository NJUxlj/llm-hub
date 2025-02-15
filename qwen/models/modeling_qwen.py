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
    def __init__(self, config: QWenConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        
        
        
        
        





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
        self.config = config
        self.layer_number = layer_idx
        self.num_expert = num_expert
        self.hidden_size = config.hidden_size
        
        
    
    
    def forward(self):
        pass
    
    
    
    





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