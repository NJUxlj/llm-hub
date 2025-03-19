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
        
        #########################
        
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
        ###############################
        
        '''
        对于以上这段判断是否训练 （is_training)的代码的详解
        
        在训练和推理时对 seqlen_k 和 seqlen_q 的不同处理，主要与注意力机制的设计和计算效率有关。以下是背后的原理：

        训练时：seqlen_k == seqlen_q
            自注意力机制：

            1. 在训练时，模型通常使用自注意力机制，即查询（q）和键（k）来自同一个输入序列。
            因此，seqlen_k 和 seqlen_q 必须相等，因为它们是同一序列的不同表示。
            因果掩码（Causal Mask）：

            2. 在训练时，通常使用因果掩码（Causal Mask），以确保每个位置只能关注到它之前的位置（包括自身）。
            如果 seqlen_k 和 seqlen_q 不相等，因果掩码无法正确应用，会导致注意力机制失效。
            
            计算效率：
            1. 训练时，序列长度通常是固定的（如填充到最大长度），因此 seqlen_k 和 seqlen_q 自然相等。
            这种一致性简化了计算，避免了额外的条件判断和调整。




        推理时：is_causal = seqlen_q == seqlen_k
            动态序列长度：
                1. 在推理时，序列长度可能是动态的（如生成式任务中逐步生成新 token）。
                因此，seqlen_k 和 seqlen_q 可能不相等，因为 seqlen_k 可能包含历史 token，而 seqlen_q 只包含当前 token。
                因果掩码的灵活性：

            推理时，is_causal 需要根据 seqlen_q 和 seqlen_k 的关系动态调整。
            如果 seqlen_q == seqlen_k，说明当前 token 和历史 token 的长度一致，可以应用因果掩码。
            如果 seqlen_q != seqlen_k，说明当前 token 和历史 token 的长度不一致，可能不需要因果掩码（如处理非自回归任务时）。
            性能优化：

            推理时，动态调整 is_causal 可以避免不必要的计算，提高效率。
            例如，如果 seqlen_q != seqlen_k，可能不需要应用因果掩码，从而减少计算量。
        
        '''
            
        
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


        self.hidden_size_per_attention_head = (
            self.projection_size // config.num_attention_heads
        )
        
        
        self.c_attn = nn.Linear(config.hidden_size, 3 * self.projection_size)
        
        self.c_proj = nn.Linear(
            config.hidden_size, self.projection_size, bias=not config.no_bias
        )
        
        
        self.is_fp32 = not (config.bf16 or config.fp16)
        
        if (
            self.use_flash_attn
            and flash_attn_unpadded_func is not None
            and not self.is_fp32
        ):
            self.core_attention_flash = FlashSelfAttention(
                causal= True, attention_dropout= config.attn_pdrop
            )
        
        self.bf16 = config.bf16
        
        
        if config.rotary_pct==1:
            self.rotary_ndims =None
        elif config.rotary_pct <1:
            self.rotary_ndims = int(self.hidden_size_per_attention_head * config.rotary_pct)
        
        dim =(
            self.rotary_ndims 
            if self.rotary_ndims is not None
            else self.hidden_size_per_attention_head,
        )
        self.rotary_emb = RotaryEmbedding(dim, base= config.rotary_emb_base)
        
        self.use_dynamic_ntk = config.use_dynamic_ntk
        self.use_logn_attn = config.use_logn_attn
        
        logn_list=[
            math.log(i, self.seq_length) if i > self.seq_length else 1
            for i in range(1, 32768)
        ]
        
        self.logn_tensor = torch.Tensor(logn_list)[None, : ,None, None]
        
        self._ntk_cached = 1
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        
    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.scale_attn_weights:
            attn_weights =  attn_weights / torch.full(
                [],
                fill_value = value.size(-1)**0.5,
                dtype = attn_weights.dtype,
                device = attn_weights.device,
            )
        query_length, key_length = query.size(-2), key.size(-2)
        '''
        # 这段代码用于创建因果注意力掩码（Causal Mask）
        # self.bias 是一个预先生成的下三角布尔矩阵（形状为 [1, 1, max_positions, max_positions]）

        # key_length: 当前key序列的总长度（包含历史信息）
        # query_length: 当前query序列的长度

        # 切片操作解析：
        # [:, :,                            -> 保留前两个维度（batch_size, head_num） 
        #   key_length - query_length : key_length  -> 在key序列维度，取与query长度对应的最后部分
        #   :key_length]                    -> 在key的序列维度取全部有效长度
        '''
        causal_mask = self.bias[
            :, :, key_length - query_length: key_length, :key_length
        ]
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full(
            [], mask_value, dtype = attn_weights.dtype
        ).to(attn_weights.device)
        
        
        attn_weights = torch.where(
            causal_mask, attn_weights.to(attn_weights.dtype), mask_value
        )
        
        attn_weights = nn.functional.softmax(
            attn_weights, dim = -1
        )
        
        attn_weights = attn_weights.type(value.dtype)
        
        attn_weights = self.attn_dropout(attn_weights)
        
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)
        attn_output = attn_output.transpose(1,2).contiguous()

        return attn_output, attn_weights
    
    def _upcast_and_reordered_attn(
        self, query, key, value, attention_mask=None, head_mask=None
    ):
        
        bsz, num_heads, q_seq_len, dk = query.size() 
        _, _, k_seq_len, _ = key.size()
        
        attention_weights = torch.empty(
            bsz* num_heads,
            q_seq_len,
            k_seq_len,
            dtype = torch.float32,
            device = query.device,
        )
        
        scale_factor = 1.0
        
        if self.scale_attn_weights:
            scale_factor = scale_factor / float(value.size(-1))  ** 0.5
        
        with autocast(enabled=False):
            q, k = query.reshape(-1, q_seq_len, dk), key.transpose(-1, -2).reshape(-1, dk, k_seq_len)
            attn_weights = torch.bmm(q, k)
            attn_weights = torch.baddbmm( # 批量矩阵乘法， 可直接看做简单的矩阵乘法
                attn_weights,  q.float(), k.float(), beta=0, alpha=scale_factor
            )
            attn_weights = attn_weights.reshape(bsz, num_heads, q_seq_len, k_seq_len)
            
            
        query_length, key_length = query.size(-2), key.size(-2)
        
        causal_mask = self.bias[:, :, key_length-query_length:key_length, :key_length ]
        
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full(
            [], fill_value=mask_value, dtype=attn_weights.dtype, device = attn_weights.device
        )
        attn_weights = torch.where(
            causal_mask, attn_weights, mask_value
        )
        
        if attention_mask is not None:
            attention_weights += attention_mask
            
            
            
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1
        )
            
        if attn_weights.dtype != torch.float32:
            raise RuntimeError(
                f"Expected attn_weights.dtype to be torch.float32, but got {attn_weights.dtype}",
                "Error with upcasting, attn_weights does not have dtype torch.float32"
            )
        
        attn_weights = attn_weights.as_type(value)
        attn_weights = self.attn_dropout(attn_weights)
            
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
            
        attn_output = torch.matmul(attention_weights, value)
        
        return attn_output, attn_weights




    def _split_heads(self, tensor, num_heads, attn_head_size):
        '''
        tensor.shape = (batch_size, seqlen, hidden_size)
        '''
        
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor
    
    def _merge_heads(self, tensor, num_heads, attn_head_size):
        '''
        tensor.shape = (batch_size, seqlen, num_heads,  attn_head_size)
        '''
        
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        tensor = tensor.view(new_shape)
        return tensor
    


    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],  
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ):
        '''
        hidden_states: shape = (batch_size, seqlen, hidden_size)
        
        '''
        mixed_x_layer = self.c_attn(hidden_states)
        query_layer, key_layer, value_layer = mixed_x_layer.split(self.split_size, dim = 2)
        
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)
        
        
        
        kv_seq_len = hidden_states.size()[1]   # 使用 input sequence size 初始化
        
        if layer_past:
            # layer past[0] shape: bs * seq_len * head_num * dim
            kv_seq_len += layer_past[0].size()[1]
            
            
        if (
            self.use_dynamic_ntk 
            and kv_seq_len == hidden_states.size()[1]
            and not self.training
        ):
            # 假设 kv_seq = 1000, seq_len = 500
            context_value = math.log(kv_seq_len/self.seq_length, 2) + 1
            ntk_alpha =  2**math.ceil(context_value) -1
            ntk_alpha = max(ntk_alpha, 1)
            self._ntk_cached = ntk_alpha
            
        else:
            ntk_alpha = self._ntk_cached    

        # 计算一个 pos-freq的矩阵 shape = (kv_seq_len, dim), 它由两个 (kv_seq_len, dim//2) 拼接而成
        rotary_pos_emb = self.rotary_emb.forward(
            kv_seq_len, ntk_alpha=ntk_alpha
        ).to(hidden_states.device)
        
        
        if rotary_pos_emb is not None:
            if isinstance(rotary_pos_emb, tuple):
                rotary_pos_emb= rotary_pos_emb
            else:
                rotary_pos_emb = (rotary_pos_emb,) * 2
                
                
                
        if rotary_pos_emb is not None:
            q_pos_emb, k_pos_emb = rotary_pos_emb
            # Slice the pos emb for current inference
            
            
            
            
        if layer_past is not None:
            past_key, past_value = layer_past[0], layer_past[1]
            
            # 拼接历史和现在
            
            key = torch.cat((past_key,key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            
        if use_cache:
            present = (key, value)
        else:
            present = None
            
        if self.use_logn_attn and not self.training:
            if self.logn_tensor.device != query.device or self.logn_tensor.dtype != query.dtype:
                self.logn_tensor = self.logn_tensor.to(query.device).type_as(query)

            seq_start = key.size()[1] - query.size()[1]
            seq_end = key.size()[1]
            logn_tensor = self.logn_tensor[:, seq_start:seq_end, :, :]
            query = query * logn_tensor.expand_as(query)


        if (
            self.use_flash_attn
            and flash_attn_unpadded_func is not None
            and not self.is_fp32
            and query.is_cuda
        ):
            q, k, v = query, key, value
            
            context_layer = self.core_attention_flash.forward(q, k, v)
            
            context_layer = rearrange(
                context_layer, "b s h d -> b s (h d)").contiguous()
        else:
            query = query.permute(0, 2, 1, 3)
            key = key.permute(0, 2, 3, 1)   
            value = value.permute(0, 2, 1, 3)
            
            attn_output, attn_weight = self._attn(
                query, key, value, attention_mask, head_mask
            )
            
            context_layer = self._merge_heads(
                attn_output, self.num_heads, self.head_dim
            )
            
        attn_output = self.c_proj(attn_output)
        outputs = (attn_output, present)
        
        if output_attentions:
            if (
                self.use_flash_attn 
                and flash_attn_unpadded_func is not None
                and not self.is_fp32    
            ):
                raise ValueError("Can not output attentions while using the flash attention~~~")
                
            else:
                outputs += (attn_weight,)

        return outputs







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
    
    
    
    





class QWenBlock(nn.Module):
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
            dim=self.hidden_size,
            eps= config.layer_norm_epsilon
        )
        
        self.attn:QWenAttention = QWenAttention(
            config = config, layer_number = layer_idx,
        )
        
        self.ln_2 = RMSNorm(
            dim=self.hidden_size,
            eps= config.layer_norm_epsilon
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

        attn_outputs = self.attn.forward(
            layernorm_output,
            layer_past = layer_past,
            attention_mask = attention_mask
        )

        
        attn_output = attn_outputs[0]
        
        outputs = attn_outputs[1:]
        
        
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
    
        layernorm_input = attn_output + residual
        
        layernorm_output = self.ln_2(layernorm_input)
        
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual  = layernorm_input
            
        mlp_output = self.mlp(layernorm_output)
        hidden_states = mlp_output + residual
        
        if use_cache:
            output = (hidden_states, ) + output
        else:
            output = (hidden_states, ) + output[1:]
        
        return outputs





class QWenPreTrainedModel(PreTrainedModel):
    config_class = QWenConfig
    base_model_prefix = "transformer"
    is_parallelizable = False
    supports_gradient_checkpointing = True
    _no_split_modules = ["QWenBlock"]
    
    def __init__(self, *inputs, **kwargs):
        super().__init__(inputs, kwargs)

    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0,
                std = self.config.initializer_range
            )
            
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0,
                std = self.config.initializer_range
            )
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        
        elif isinstance(module, RMSNorm):
            module.weight.data.fill_(1.0)
        
        for name, p in module.named_parameters():
            if name =="c_proj.weight":
                p.data.normal_(
                    mean=0.0,
                    std = (
                        self.config.initializer_range
                        / math.sqrt(2 * self.config.n_layers)
                    )
                )

    
    def _set_gradient_checkpointing(self, module, value = False):
        if isinstance(module, QWenModel):
            module.gradient_checkpointing = value
        
    
    
    
class QWenModel(QWenPreTrainedModel):
    # _keys_to_ignore_on_load_missing：这是一个类属性，用于定义在加载模型权重时，哪些键（key）可以忽略缺失。
    # 如果模型权重文件中缺少 attn.masked_bias 这个参数，加载时不会报错，而是直接跳过。
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]
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
            self.init_method(self.position_embeddings.weight)
            self._position_embeddings_key = "position_embeddings"
            self.init_method(self.position_embeddings.weight)
            
        else:
            self.wpe = None
            self._position_embeddings_key = ""
            
            
        self.wte = nn.Embedding(self.vocab_size, self.embed_dim)
        
        self.dropout = nn.Dropout(config.embd_pdrop)
        
        self.h = nn.ModuleList(
            [
                QWenBlock(
                    config,
                    layer_idx=i
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        
        
        self.ln_f = RMSNorm(
            self.embed_dim,
            eps = config.layer_norm_epsilon
        )
        
        self.post_init()
        
        
    def get_input_embeddings(self):
        return self.wte
    
    
    def set_input_embedding(self, value):
        self.wte = value
    
        
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
            
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_shape[0]
        
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = input_shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        
        
        if token_type_ids is not None:
             token_type_ids = token_type_ids.view(-1, input_shape[-1])
         
        if position_ids is not None:
             position_ids = position_ids.view(-1, input_shape[-1])
         
         
        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))  # 长度为num_hidden_layers（模型总层数）的元组
        else:
            past_length = past_key_values[0][0].size(-2)
        
        
        
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                past_length + input_shape[-1],
                dtype = torch.long,
                device = device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        
        
        
        if attention_mask is not None:
            if batch_size <=0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:,None,None,:]

            attention_mask  = attention_mask.to(dtype=self.dtype)
            '''
            1 1 1 0
            1 1 1 0
            1 1 1 0
            
            置返
            
            0 0 0 1
            0 0 0 1
            0 0 0 1
            
            1 变为 -无穷
            0 0 0 -inf
            0 0 0 -inf
            0 0 0 -inf
            '''
            
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
            
        
        encoder_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.n_layers)
        
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
            
        hidden_states = inputs_embeds
        
        if self.wpe is not None:
            posiiton_embeds = self.wpe(position_ids)
            hidden_states = hidden_states + posiiton_embeds
            
        hidden_states = self.dropout(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),) # shape = [batch_size, seqlen, hidden_size]
        
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
                
                
                
        presents= () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        
        
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module:QWenBlock):
                    def custom_forward(*inputs):
                        return module.forward(inputs, use_cache, output_attentions)
                    
                    return custom_forward
                
                
                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
                
            else:
                block:QWenBlock
                outputs = block.forward(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
                
  
            hidden_states = outputs[0]
            
            if use_cache:
                presents = presents + (outputs[2 if output_attentions else 1], )
                
                
            if output_attentions:
                all_self_attentions += (outputs[1],)
        
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(output_shape)
        
        if not return_dict:
            return tuple(
                v for v in [hidden_states, presents, all_hidden_states] if v is not None
            )
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )




class QWenLMHeadModel(QWenPreTrainedModel):
    
    _keys_to_ignore_on_load_missing = [r"h\.\d+\.attn\.rotary_emb\.inv_freq"]
    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.masked_bias"]

    
    def __init__(self, config: QWenConfig):
        super().__init__(config)
        assert (
            config.bf16+config.fp16+config.fp32 <=1
        ), "Only one of \"bf16\", \"fp16\", \"fp32\" can be true"

        autoset_precision = config.bf16 + config.fp16 + config.fp32 
        
        if autoset_precision:
            pass
        
        if config.bf16 and SUPPORT_CUDA and not SUPPORT_BF16:
            logger.warn(
                "Your device does NOT seem to support bf16, you can switch to fp16 or fp32 by by passing fp16/fp32=True in \"AutoModelForCausalLM.from_pretrained\"."
            )
        
        if config.fp16 and SUPPORT_CUDA and not SUPPORT_FP16:
            logger.warn(
                "Your device does NOT seem to support fp16, you can switch to fp32 by by passing fp32=True in \"AutoModelForCausalLM.from_pretrained\"."
            )
            
        if config.fp32:
            if SUPPORT_BF16:
                pass
            elif SUPPORT_FP16:
                pass
            
        if config.use_flash_attn == "auto":
            pass
        
        
        if config.use_flash_attn and config.fp32:
            logger.warn(
                
            )
            
        if config.use_flash_attn:
            global apply_rotary_emb_func, rms_norm, flash_attn_unpadded_func
            
            try:   
                from flash_attn.layers.rotary import apply_rotary_emb_func as __apply_rotary_emb_func 
                apply_rotary_emb_func  =__apply_rotary_emb_func
            except ImportError as e:
                logger.warn(
                    "Warning: import flash_attn rotary fail, please install FlashAttention rotary to get higher efficiency "
                    "https://github.com/Dao-AILab/flash-attention/tree/main/csrc/rotary"
                )
        
            try:
                from flash_attn.ops.rms_norm import rms_norm as __rms_norm
                rms_norm = __rms_norm
            except ImportError:
                logger.warn(
                    "Warning: import flash_attn rms_norm fail, please install FlashAttention layer_norm to get higher efficiency "
                    "https://github.com/Dao-AILab/flash-attention/tree/main/csrc/layer_norm"
                )
                
                
            try:
                from flash_attn.flash_attn_interface import flash_attn_unpadded_func as __flash_attn_unpadded_func
                flash_attn_unpadded_func = __flash_attn_unpadded_func
            except ImportError:
                logger.warn(
                    "Warning: import flash_attn fail, please install FlashAttention to get higher efficiency "
                    "https://github.com/Dao-AILab/flash-attention"
                )
        
        
        
        
        self.transformer:QWenModel = QWenModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
    
        if config.bf16:
            self.transformer.bfloat16()
            self.lm_head.bfloat16()
        if config.fp16:
            self.transformer.half()
            self.lm_head.half()
   
        self.post_init()
        
        
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1) # 取最后一个token
            
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)
        
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)
        
        # 从atterntion_mask 中推导出position_ids
        if attention_mask is not None and position_ids is not None:
            position_ids = attention_mask.long().cumsum(-1)-1
            position_ids.masked_fill(attention_mask==0, 1)
            if past_key_values:
                position_ids = position_ids[:,-1].unsqueeze(-1)
        else:
            position_ids = None
            
            
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}  
        else:
            model_inputs = {"input_ids": input_ids}
            
            
        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,   
        })
        
        
        return model_inputs
        
    
    
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        
        transformer_outputs = self.transformer.forward(
            input_ids = input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = transformer_outputs[0]
        
        lm_logits = self.lm_head(hidden_states) # 映射到词表大小
        
        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            shift_logits = lm_logits[... , :-1,:].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(), shift_labels.view()
            )
        
        if not return_dict:
            output = (lm_logits, ) + transformer_outputs[1:]
            return ((loss, ) + output) if loss is not None else output
         
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
        
        
        
        
    @staticmethod
    def _reorder_cache():
        pass
    
    
    
    
    def chat(self):
        pass
    
    
    
    
    
    def generate(self):
        pass







class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.inv_freq = 1.0 /(base**(torch.arange(0, dim, 2).float()/dim))
        
        if importlib.util.find_spec("einops") is None:
            raise RuntimeError("einops is required for Rotary Embedding")
        
        
        self._rotary_emb_pos_cache = None
        self._seq_len_cached = 0
        self._ntk_alpha_cached = 1.0
        
    def update_rotary_pos_emb_cache(self, max_seq_len, offset=0, ntk_alpha=1.0):
        '''
        ##参数详解：

        max_seq_len（核心参数）：
            当前处理的最大序列长度
            决定需要生成的位置编码的长度
            典型应用：处理输入序列时根据实际长度动态调整
        offset=0（扩展参数）：
            序列起始偏移量
            用于处理分段序列或缓存复用场景
            示例：当处理第1000-2000个token时，offset=1000
        ntk_alpha=1.0（NTK关键参数）：
            Neural Tangent Kernel缩放系数
            用于动态调整基值(base)实现上下文长度外推
            >1时扩展模型上下文窗口，<1时收缩窗口

            典型值范围：1.0-4.0（根据任务调整）
        
        
        技术特点：
            动态缓存机制：只在序列长度超过缓存或参数变化时重新计算，提升效率
            NTK-aware编码：通过ntk_alpha实现无需训练的长度外推
            复数形式编码：通过cat操作实现复数表示（cosθ + i*sinθ）
            设备一致性：保持计算设备与原始参数一致（GPU/CPU）
        
        典型应用场景：
            处理超长文本输入时自动扩展上下文窗口
            流式处理中动态更新位置编码
            复用已有位置编码缓存提升推理速度
            该方法属于旋转位置编码的优化实现，通过动态调整编码参数来平衡计算效率和模型性能。
        '''
        # seqlen = 当前需要处理的序列长度 = 最大序列长度 + 偏移量
        seqlen  =  max_seq_len + offset
        # 当遇到更长的序列或缩放系数变化时更新缓存
        if seqlen > self._seq_len_cached and ntk_alpha != self._ntk_alpha_cached:
            # 应用NTK-aware缩放后的基值计算
            base = base * ntk_alpha**(self.dim/(self.dim-2))
             # 重新计算频率倒数（核心的旋转位置参数）
            inv_freq = 1.0 / (
                base **(torch.arange(0, self.dim, 2, device=self.inv_freq.device).float()/self.dim)
            )
            
            # 更新缓存记录值
            self._seq_len_cached = seqlen
            self._ntk_alpha_cached = ntk_alpha
            seq = torch.arange(seqlen, device=self.inv_freq.device) # 创建一个position'序列
            # 构造一个从 pos-> freqs的矩阵
            freqs = torch.outer(seq.type_as(self.inv_freq), self.inv_freq)
            
            # 拼接实部和虚部（形成复数表示）
            emb = torch.cat((freqs, freqs), dim=-1)  # shape = [seqlen, dim]
            
            from einops import arrange
            
            self._rotary_emb_pos_cache = rearrange(emb, "n d -> 1 n 1 d")
            
    
    
    
    def forward(self, max_seq_len, offset=0, ntk_alpha=1.0):
        self.update_rotary_pos_emb_cache(max_seq_len, offset, ntk_alpha)
        return self._rotary_emb_pos_cache[:,offset: offset+max_seq_len]
        
        
        


def _rotate_half(x: torch.Tensor):
    from einops import rearrange # 动态导入张量重塑库
    
    # 将输入张量的最后一个维度分割为2个子维度
    # 例如：形状从 [..., 256] -> [..., 2, 128]
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    
    # 沿倒数第二个维度解绑张量，得到两个子张量
    # x1形状：[..., 128], x2形状：[..., 128]
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)
    




def apply_rotary_pos_emb(t, freqs):
    '''
    确定旋转维度：从 freqs 张量的最后一个维度获取旋转维度 rot_dim。
    分割输入张量：将输入张量 t 分割为两部分：t_ 包含前 rot_dim 个元素，t_pass_ 包含剩余元素。
    转换数据类型：将 t_ 和 t_pass_ 转换为浮点类型。
    
    应用旋转位置编码：对 t_ 应用旋转位置编码公式：
    首先，计算 freqs 的余弦值，并与 t_ 逐元素相乘。
    然后，调用 _rotate_half 函数对 t_ 进行半旋转操作，并计算 freqs 的正弦值，将两者逐元素相乘。
    最后，将上述两个结果相加，得到编码后的 t_。
    合并张量：将编码后的 t_ 和 t_pass_ 沿着最后一个维度拼接起来，得到最终的编码结果。
    
    '''
    if apply_rotary_emb_func is not None:
        _t = t 
        freqs = freqs.squeeze(0).squeeze(1) # shape = [seqlen, dim]
        cos = freqs[:, :freqs.shape[-1]//2].cos()   # freqs.shape[-1] == dim
        sin = freqs[:, freqs.shape[-1]//2:].sin()
        output =  apply_rotary_emb_func(_t, cos, sin).type_as(t)
        
        return output
    
    else:
        rot_dim = freqs.shape[-1]
        t_, t_pass = t[..., :rot_dim], t[..., rot_dim:]
        t_ = t_.float()
        t_pass = t_pass.float()
        t_ = (t_ * freqs.cos()) + (_rotate_half(t_)* freqs.sin())
        return torch.cat([t_, t_pass],dim=-1).type_as(t)


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight =  nn.Parameter(torch.ones(dim))
        
        
    def _norm(self, x):
        rms_x = torch.sqrt(torch.square(x).mean(-1, keepdim=True), dim=-1)
        
        output= torch.divide(x, rms_x+ self.eps) # 避免除零错误
        
        return output

    
    def forward(self, x):
        if rms_norm is not None and x.is_cuda:
            return rms_norm(x, self.weight, self.eps)
        else:
            output = self._norm(x.float()).type_as(x)
            
        return output * self.weight