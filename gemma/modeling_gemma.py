# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch Gemma model."""


import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_causal_attention_mask,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_gemma import GemmaConfig


if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

'''
index_first_axis：用于对张量的第一维进行索引操作。
pad_input：用于对输入序列进行填充，使其长度一致。
unpad_input：用于去除填充的部分，恢复原始序列。
'''

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

'''
使用 torch.fx.wrap 函数将 _prepare_4d_causal_attention_mask 函数包装起来，使其成为FX图中的叶子节点。
这样，在进行图追踪时，该函数将作为一个整体节点出现在图中，而不会被进一步拆解。
'''


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "GemmaConfig"



def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * (1 + self.weight)


ALL_LAYERNORM_LAYERS.append(GemmaRMSNorm)




class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base # theta
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, input_ids, seq_len=None):
        
        
        if self.inv_freq is None:
            pass
        


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    
    '''
    功能：
        该函数将输入张量 x 的最后一个维度的元素进行旋转操作，即将最后一个维度的后半部分元素取负后，与前半部分元素拼接在一起。
    代码详解：
        x1 = x[..., :x.shape[-1]//2]：从输入张量 x 中取出最后一个维度的前半部分元素。
        x2 = x[...,x.shape[-1]//2:]：从输入张量 x 中取出最后一个维度的后半部分元素。
        return torch.cat([-x2,x1], dim=-1)：将后半部分元素取负后，与前半部分元素在最后一个维度上进行拼接。
    '''
    
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[...,x.shape[-1]//2:]
    
    return torch.cat([-x2,x1], dim=-1)



# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.


    Function:
        功能：
            该函数将旋转位置编码应用到查询（query）和键（key）张量上，通过使用余弦（cos）和正弦（sin）部分的旋转嵌入，对查询和键进行旋转操作。
        代码详解：
            cos = cos.unsqueeze(unsqueeze_dim) 和 sin = sin.unsqueeze(unsqueeze_dim)：在指定的维度上对余弦和正弦张量进行维度扩展，以便能够与查询和键张量进行广播操作。
            q_embed = (q * cos) + (rotate_half(q) * sin)：对查询张量 q 应用旋转位置编码，通过将查询张量与余弦张量相乘，再加上旋转后的查询张量与正弦张量的乘积。
            k_embed = (k * cos) + (rotate_half(k) * sin)：对键张量 k 应用旋转位置编码，同样通过将键张量与余弦张量相乘，再加上旋转后的键张量与正弦张量的乘积。
            return q_embed, k_embed：返回经过旋转位置编码后的查询和键张量。
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Gemma
class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    
    
    
    
    





# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    
    
    



class GemmaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Ignore copy
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )