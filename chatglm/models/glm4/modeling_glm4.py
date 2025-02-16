# coding=utf-8
# Copyright 2024 The GLM & ZhipuAI team and HuggingFace Inc. team. All rights reserved.
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

import torch
from torch import nn

from typing import Optional, List, Tuple, Union, Any, Callable, Dict, Type

import torch.utils.checkpoint


from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.generation import GenerationMixin
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.utils.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.utils.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils.processing_utils import Unpack
from transformers.utils import (
    LossKwargs,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg
from .configuration_glm4 import GlmConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "THUDM/glm-4-9b"
_CONFIG_FOR_DOC = "GlmConfig"


class GlmMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.activation_fn = ACT2FN[config.hidden_act]
        
    def forward(self,hidden_states:torch.Tensor)->torch.Tensor:
        up_states = self.gate_up_proj(hidden_states)
        
        gate, up_states = up_states.chunk(2, dim=-1)
        
        up_states = self.activation_fn(gate) * up_states
        
        down_states = self.down_proj(up_states)

        return down_states



# class GLMBlock(nn.Module):
#     def __init__(self, config: ChatGLMConfig, layer_number, device=None):
#         super().__init__()
#         self.config = config
#         self.hidden_size = config.hidden_size
#         self.num_attention_heads = config.num_attention_heads




def repeat_kv():
    pass






def eager_attention_forward():
    pass




def rotate_half():
    pass





def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    pass









class GlmAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config:GlmConfig, layer_idx:Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx 
        self.head_dim = getattr(config,"head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        
        self.is_casual = True
        
        
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        
        self.v_proj = nn.Linear(
           config.hidden_size, config.num_key_value_heads* self.head_dim, bias=config.attention_bias 
        )
        
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=False
        )
        
        
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: Tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor]=None,
            **kwargs: Unpack[FlashAttentionKwargs]
        )->Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        '''
        hidden_states.shape = (batch_size, seq_length, hidden_size)
        '''
        input_shape = hidden_states.shape[:-1] # shape = (batch_size, seq_length)
        hidden_shape = (*input_shape, -1, self.head_dim) # shape = (batch_size, seq_length, num_attention_heads, head_dim)

        # self.q_proj(hidden_states).shape  = (batch_size, seq_length, num_attention_heads * head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1,2) # shape = (batch_size, num_heads, seq_length, head_dim)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1,2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1,2)

        cos, sin = position_embeddings # shape = 
        
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )
        
        
        if past_key_value is not None:
            cache_kwargs = {"sin":sin, "cos":cos, "cache_position":cache_position}
            
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx,**cache_kwargs
            )

        
        attention_interface:Callable = eager_attention_forward
    
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
                logger.warning_once(
                    "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                    'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
                )
            
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.transpose(1,2).contiguous() # shape = (batch_size, seq_length, num_attention_heads * head_dim)
        attn_output = self.o_proj(attn_output) # shape = (batch_size, seq_length, hidden_size)
        
        return attn_output, attn_weights
    
    
    

class GlmRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states):
        pass
        
    
    
    


class GlmRotaryEmbedding(nn.Module):
    def __init__(self, config:GlmConfig, device=None):
        super().__init__()

    
