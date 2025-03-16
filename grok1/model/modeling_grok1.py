from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

try:
    from transformers.modeling_attn_mask_utils import \
        _prepare_4d_causal_attention_mask

    HAS_MASK_UTILS = True
except ImportError:
    HAS_MASK_UTILS = False

from .configuration_grok1 import Grok1Config
from .modeling_grok1_outputs import (MoeCausalLMOutputWithPast,
                                     MoeModelOutputWithPast)

logger = logging.get_logger(__name__)



# copied from https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/models/mixtral/modeling_mixtral.py
def load_balancing_loss_func(
    gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2
) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of tensors. Shape: [batch_size, seqeunce_length, num_experts].
        num_experts (`int`, *optional*):
            Number of experts

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None:
        return 0

    if isinstance(gate_logits, tuple):
        # cat along the layers?
        compute_device = gate_logits[0].device
        gate_logits = torch.cat(
            [gate.to(compute_device) for gate in gate_logits], dim=0
        ) # shape = [batch_size * num_layers, seqeunce_length, num_experts]

    routing_weights, selected_experts = torch.topk(gate_logits, top_k, dim=-1)
    routing_weights = routing_weights.softmax(dim=-1) # 
    
    # 输出形状：
    # routing_weights.shape == selected_experts.shape == [batch*num_layers, seq_len, top_k]

 
    # cast the expert indices to int64, otherwise one-hot encoding will fail
    if selected_experts.dtype != torch.int64:
        selected_experts = selected_experts.to(torch.int64)

    if len(selected_experts.shape) == 2:  # top_k == 1
        selected_experts = selected_experts.unsqueeze(2) # shape = [batch_size * num_layers, seqeunce_length, 1]

    # router_logits 经过 topk 操作后得到 selected_experts 是专家索引 (batch*layers, seq_len, top_k)
    # one_hot 将其转换为三维掩码矩阵 (batch*layer, seq_len, top_k, num_experts)
    # 每个位置表示该位置的专家是否被选中
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
    
    # 例如当 selected_experts = [[0,2], [1,3]]，num_experts=4 时：
    # expert_mask 会变成：
    # [[[1,0,0,0], [0,0,1,0]],
    #  [[0,1,0,0], [0,0,0,1]]]

    # For a given token, determine if it was routed to a given expert.
    # 步骤1：合并top_k维度的专家选择结果（每个token可能选择多个专家）
    # 输入形状：[batch*num_layers, seq_len, top_k, num_experts]
    # 输出形状：[batch*num_layers, seq_len, num_experts]
    expert_mask = torch.max(expert_mask, axis=-2).values
    
    # max操作合并top_k维度后：  
    [[1,0,1,0],  
    [0,1,0,1]]  

    # cast to float32 otherwise mean will fail
    expert_mask = expert_mask.to(torch.float32) # [batch*num_layers, seq_len, num_experts]

    # 步骤3：计算 对于整个序列，每个专家被选中的平均概率
    # 沿序列维度（seq_len）求平均，结果形状：[batch*num_layers, num_experts] = [B, E]
    tokens_per_group_and_expert = torch.mean(expert_mask, axis=-2)

    # 步骤4：计算路由权重的平均概率 【每个token选择top-k专家的平均概率】
    # 输入形状：[batch*num_layers, seq_len, top_k]
    # 结果形状：[batch*num_layers, seq_len] = [B, S]
    router_prob_per_group_and_expert = torch.mean(routing_weights, axis=-1)
    
    # 最终通过二者的点乘计算负载均衡损失： 
        # $$Loss = E[P_{token}(e) \cdot P_{router}(e)] \cdot N_{experts}^2$$ 
        # 这惩罚了专家选择分布与路由权重分布不一致的情况。
        # e 表示 专家索引 0~E-1
        # P_{token}(e) 表示 专家e被选中的概率
        # P_{router}(e) 表示 路由选择到专家e的概率
        
    # [B,E] * [B,S,1] -> [B,1,E] * [B,S,1] -> [B,S,E] -> 标量
    
    
    '''
    # 通过unsqueeze(-1)添加新维度
    router_prob_per_group_and_expert.unsqueeze(-1).shape = [B, S, 1]

    # 广播机制生效后的相乘过程：
    [B, E]          # tokens_per_group
        × 
    [B, S, 1]       # router_prob
    → 
    [B, S, E]       # 广播后的乘积结果

    # 最终求均值时的维度：
    torch.mean([B, S, E]) → 标量
    
    这里能够相乘的关键在于PyTorch的广播机制（Broadcasting），具体规则如下：
        维度对齐：从右向左对齐
        [B, E] → 实际维度为 [B, 1, E]
        [B, S, 1] → 保持原状
    
    这种设计实际上计算的是专家选择概率与路由概率在所有位置（sequence positions）和所有专家（experts）上的期望乘积，
        最终通过乘以专家数的平方来放大不平衡惩罚。
    '''
    return torch.mean(
        tokens_per_group_and_expert * router_prob_per_group_and_expert.unsqueeze(-1)
    ) * (num_experts**2)




# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)





class RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        create_scale: bool = True,
    ) -> None:
        super().__init__()
        self.variance_epsilon = eps
        if create_scale:
            self.scale = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.scale = 1.0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.scale * hidden_states
        return hidden_states.to(input_dtype)
    
    



class RotaryEmbedding(nn.Module):
    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: int = 10000
    ) -> None:
        super().__init__()
        assert dim % 2 == 0, f'hidden_dim must be divisible by 2, but get {dim}'
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        # 构造一个pos_theta矩阵
        freqs = torch.outer(t, self.inv_freq) # shape = (seq_len, dim/2)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)  # shape = (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached: 
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )
        
        
    
# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """
    Rotates half the hidden dims of the input.
    
    ###Code
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)
    
    ### Return
    return torch.tensor shape = [batch_size, seq_len, hidden_size]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)





# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding. shape = [batch_size, max_seq_len, head_dim]
        sin (`torch.Tensor`): The sine part of the rotary embedding.  shape = [batch_size, max_seq_len, head_dim]
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    

    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # 扩展到q,k 的对应形状
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    
    # q*cos: 将查询向量与余弦位置编码逐元素相乘
    # rotate_half(q)*sin: 先将查询向量旋转一半维度，再与正弦位置编码逐元素相乘
    q_embed = (q*cos) + (rotate_half(q)*sin)
    k_embed = (k*cos) + (rotate_half(k)*sin)
    return q_embed, k_embed



class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: Optional[int] = None,
        max_position_embeddings: int = 2048,
        attn_output_multiplier: float = 1.0,
        max_attn_val: float = 30.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        if num_key_value_heads is None:
            num_key_value_heads = num_heads  # # kv_channels
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attn_output_multiplier = attn_output_multiplier
        self.max_attn_val = max_attn_val

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        
        self.q_proj = nn.Linear(hidden_size, self.num_heads * self.head_dim, bias=False)
        
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=False )

        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads* self.head_dim, bias=False)

        self.o_proj = nn.Linears(self.num_heads*self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings = max_position_embeddings,
        )
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)  
        
        
        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        
        
        kv_seq_len = key_states.shape[-2]

        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

        
        cos, sin = self.rotary_emb.forward(value_states, seq_len = kv_seq_len)
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        
        
        # 将历史记录分别拼接到KV矩阵上
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim = 2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)


        # 更新本轮的past_key_value
        past_key_value = (key_states, value_states) if use_cache else None
        
        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups) # 一个 group里面包含多个 key_value heads
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)).to(torch.float)
        
        attn_weights = attn_weights * self.attn_output_multiplier
        attn_weights = self.max_attn_val * F.tanh(attn_weights/self.max_attn_val)
        
        
        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
            
        if attention_mask is not None:
            if attention_mask.size()!=(bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
        
            attn_weights = attn_weights + attention_mask
        
        
        attn_weights = F.softmax(attn_weights, dim=-1).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"attn_output should be size {(bsz, self.num_heads,q_len, self.head_dim)}", 
                f"but is {attn_output.size()}"
            )
            
        attn_output = attn_output.transpose(1,2).contiguous()
        attn_output = attn_output.reshape((bsz, q_len, self.hidden_size))
        
        attn_output =  self.o_proj(attn_output)
        
        if not output_attentions:
            attn_weights = None
            
        return attn_output, attn_weights, past_key_value


class MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
    ) -> None:
        super().__init__()
        self.linear_v = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.linear_1 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.linear = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        current_hidden_states = self.act_fn(self.linear(hidden_states)) * self.linear_v(
            hidden_states
        )
        current_hidden_states = self.linear_1(current_hidden_states)
        return current_hidden_states
    
    







class MoeBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [MoeMLP(hidden_dim, ffn_dim) for _ in range(num_experts)]
        )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        
        # router_logits: (batch * sequence_length, n_experts)
    
        router_logits = self.gate(hidden_states)
        routing_weights =  F.softmax(router_logits, dim=-1, dtype = torch.float)    

        routing_weights, selected_experts = torch.topk(
            routing_weights, self.top_k, dim=-1
        ) # (batch * sequence_length, top_k), ()
        
        
        routing_weights =  routing_weights.to(hidden_states.dtype)
    
    
    
class DecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_key_value_heads: int,
        num_experts: int,
        top_k: int,
        max_position_embeddings: int = 2048,
        attn_output_multiplier: float = 1.0,
        max_attn_val: float = 30.0,
        rms_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(
            hidden_size,
            num_heads,
            num_key_value_heads,
            max_position_embeddings=max_position_embeddings,
            attn_output_multiplier=attn_output_multiplier,
            max_attn_val=max_attn_val,
        )
        self.moe_block = MoeBlock(hidden_size, intermediate_size, num_experts, top_k)
        self.pre_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attn_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.pre_moe_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_moe_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        
        
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        '''
        ## Args:
            hidden_states: (batch_size, sequence_length, hidden_size)。
        '''
        residual = hidden_states
        hidden_states = self.pre_attn_norm(hidden_states)
        hidden_states, attention_weights, present_key_value = self.attn.forward(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = self.post_attn_norm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_moe_norm(hidden_states)
        
        '''
        router_logits 是在混合专家（Mixture of Experts, MoE）模块中，
            由门控网络（gate network）输出的对数概率值，
            用于决定每个输入样本应该路由到哪些专家网络。
            其形状通常为 (batch_size, sequence_length, num_experts)。
        '''
        hidden_states, router_logits = self.moe_block(hidden_states)
        hidden_states = self.post_moe_norm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attention_weights,)
        if use_cache:
            outputs += (present_key_value,)
        if output_router_logits:
            outputs += (router_logits,)
        return outputs
    
    
    
    



class Grok1PretrainedModel(PreTrainedModel):
    config_class = Grok1Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_cache_class = False

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.zero_()
            
            

# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """

    bsz, tgt_len = input_ids_shape # tgt_len = query_size
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    
    # 创建因果注意力掩码，其核心作用是生成一个下三角矩阵，确保自注意力机制只能关注当前位置及之前的token
    # mask_cond 代表行， (mask_cond+1.view(-1, 1)) 代表列
    '''
    mask_cond = torch.tensor([0, 1, 2])  # 位置索引

    # 表达式分解步骤：
    right_matrix = (mask_cond + 1).view(3, 1)  
    # 得到 [[1], [2], [3]]

    comparison = mask_cond < right_matrix  
    # 广播比较生成3x3布尔矩阵：
    # [[0<1, 0<2, 0<3],   → [True, True, True]
    #  [1<1, 1<2, 1<3],   → [False, True, True]
    #  [2<1, 2<2, 2<3]]   → [False, False, True]
    
    当执行 mask.masked_fill_(comparison, 0) 后：

    [[ 0,    0,    0],   # 可以关注所有位置
    [-inf,  0,    0],   # 只能关注位置1及之后
    [-inf, -inf,  0]]   # 只能关注位置2
    这实现了：

    对角线及下方区域填充0（允许关注）
    上方区域保持-inf（禁止关注）
    '''
    
    mask.masked_fill(mask_cond <(mask_cond+1).view(mask.size(-1), 1), 0)
    
    # 修正后的比较条件：
    # mask.masked_fill(mask_cond[:, None] >= (mask_cond[None, :] + 1), 0)
    
    mask = mask.to(dtype)
    
    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask, # shape = (tgt_len, tgt_len)
            ]
            ,dim=-1
            ) # shape = (tgt_len, tgt_len + past_key_values_length)
        
    '''
    假设 tgt_len ==3, past_len =1 拼接完毕后就是:
        [[0, 0,    0,    0],   # 可以关注所有位置
        [0, -inf,  0,    0],   # 只能关注位置1及之后
        [0, -inf, -inf,  0]]   # 只能关注位置2 
    
    '''
    return mask[None,None,:,:].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )



# Copied from transformers.models.bart.modeling_bart._expand_mask

def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    
    src_len（源序列长度）
        指当前输入的序列长度（不含历史信息），即attention_mask的原始维度 [bsz, src_len] 中的第二个维度

    tgt_len（目标序列长度）
        指要生成的序列长度（解码器侧），默认与src_len相同。当需要处理不同长度的输入输出时（如机器翻译），两者会不同

    与past_key_value_length的关系
        这两个参数不包含past_key_value_length
    
    mask： padding掩码，举例：

    假设原始mask为：

        mask = torch.tensor([[1, 1, 0]])  # bsz=1, src_len=3
        tgt_len = 4
        dtype = torch.float32
        
        经过代码处理后：
        expanded_mask = mask[:, None, None, :]  # shape [1, 1, 1, 3]
                .expand(1, 1, 4, 3)        # shape [1, 1, 4, 3]
                .to(torch.float32)
        
        最终得到四维张量：
        # 形状：[1, 1, 4, 3]
        tensor([[
            [
                [1.0, 1.0, 0.0],  # 目标位置1的掩码
                [1.0, 1.0, 0.0],  # 目标位置2的掩码
                [1.0, 1.0, 0.0],  # 目标位置3的掩码
                [1.0, 1.0, 0.0]   # 目标位置4的掩码
            ]
        ]], dtype=torch.float32)
        
        后续经过inverted_mask = 1.0 - expanded_mask和掩码填充后，会变成：
        tensor([[
            [
                [0.0, 0.0, -inf],  # 目标位置1的最终掩码
                [0.0, 0.0, -inf],  # 目标位置2的最终掩码 
                [0.0, 0.0, -inf],  # 目标位置3的最终掩码
                [0.0, 0.0, -inf]   # 目标位置4的最终掩码
            ]
        ]])
        
        这样每个目标位置都会：

        看到完整的源序列（src_len=3）
        对填充位置（第3列）应用-inf
    """
    
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len   # tgt_len == query_size
    
    expanded_mask = mask[:,None,None,:].expand(bsz, 1, tgt_len, src_len).to(dtype)
    
    # 将原来的掩码取反
    '''
    将原始掩码值反转（1变0，0变1），因为：
    原始掩码中的1表示有效token，0表示填充token
    转换后，填充位置会被赋予极大负值（在softmax后趋近于0）
    '''
    inverted_mask = 1 - expanded_mask
    
    # 用极小值填充掩码区域（使softmax后接近0）
    return inverted_mask.masked_fill(
        
    )


class Grok1Model(Grok1PretrainedModel):
    def __init__(self, config: Grok1Config, **kwargs) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embedding_multiplier_scale = config.embedding_multiplier_scale
        
        
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        
        self.layers = nn.ModuleList([
            DecoderLayer(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_attention_heads,
                num_key_value_heads=config.num_key_value_heads,
                num_experts=config.num_experts,
                top_k=config.num_experts_per_tok,
                max_position_embeddings=config.max_position_embeddings,
                attn_output_multiplier=config.attn_output_multiplier,
                max_attn_val=config.max_attn_value,
                rms_norm_eps=config.rms_norm_eps,
            )
            for i in range(config.num_hidden_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.gradient_checkpoint = False
        self.post_init()
        
    
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
        
        
        
    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ): 
        '''
        这段代码的主要作用是准备解码器的注意力掩码(attention mask)，具体功能如下：

        创建因果掩码(Causal Mask)：
            当输入序列长度大于1时，使用_make_causal_mask函数创建一个下三角形式的因果掩码
            这种掩码确保每个token只能关注它自身及之前的token，防止信息泄露
        
        处理输入注意力掩码：
            如果传入了attention_mask，使用_expand_mask函数将其扩展到适合注意力机制的四维形状
            扩展后的掩码形状为[batch_size, 1, target_seq_len, source_seq_len]
        
        合并掩码：
            将因果掩码和扩展后的注意力掩码进行合并
            如果两者都存在，则进行相加操作
        
        返回结果：
            最终返回一个适合解码器使用的组合注意力掩码
        '''
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        
        combined_attention_mask = None
        if input_shape[-1] >1: # 如果输入序列长度大于1
            combined_attention_mask = _make_causal_mask(  # 历史记录对应的掩码
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )
    
        
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(  # 当前输入序列对应的掩码
                attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            
            combined_attn_mask = (
                expanded_attn_mask 
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask,
            )
            
        return combined_attention_mask
    
    
    
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    )->Union[Tuple, MoeModelOutputWithPast]:
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
        
        
        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds"
            )
            
        seq_length_with_past = seq_length
        past_key_values_length = 0
        
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length
        
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            
            position_ids = position_ids.unsqueeze(0)
        
        
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            inputs_embeds = inputs_embeds * self.embedding_multiplier_scale
        
        
        if HAS_MASK_UTILS:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            if attention_mask is None:
                attention_mask = torch.ones(
                    (batch_size, seq_length_with_past),
                    dtype = torch.bool,
                    device = inputs_embeds.device
                )
                
            attention_mask = self._prepare_decoder_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        
        # embed positions
        hidden_states = inputs_embeds
        
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
                
                
                
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = () if use_cache else None
        

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_values = (
                past_key_values[idx] if past_key_values is not None else None
            )
            
            if self.gradient_checkpointing and self.training:
                
            
                def create_custom_forward(module:nn.Module):
                    def custom_forward(*inputs):
                        return module.forward(*inputs, past_key_values, output_attentions)

                    return custom_forward
            
                layer_outputs =torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                )
            
            else:  # 没有梯度检查点的正常前向计算
                layer_outputs = decoder_layer.forward(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states  = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1] , )
                
            if output_attentions:
                all_self_attns += (layer_outputs[1],)
            
            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)
            
        
        # 所有decoder都走完了
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
            
        next_cache = next_decoder_cache if use_cache else None

        
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_router_logits,
                ]
                if v is not None
            )
        
        else:
            return MoeModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=next_cache,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
                router_logits=all_router_logits,
            )

    
    
    





class Grok1ModelForCausalLM(Grok1PretrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    
    
    
    def __init__(self, config: Grok1Config, **kwargs):
        super().__init__(config)
        self.model:Grok1Model = Grok1Model(config)
        self.vocab_size = config.vocab_size
        self.output_multiplier_scale = config.output_multiplier_scale
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()
        
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
        
        
        
    def get_output_embeddings(self):
        return self.lm_head
    
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        
        
    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    
    
    
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    )->Union[Tuple, MoeCausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_router_logits = (
            output_router_logits
            if output_router_logits is not None
            else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
        )
        
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states) * self.output_multiplier_scale

        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()   # "今天天气很" 
            shift_labels = labels[..., 1:].contiguous()       # "天天气很好"
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size) # shape = (bsz * seqlen, vocab_size)
            shift_labels = shift_labels.view(-1)   # shape = (bsz * seqlen)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            
        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits, # 所有layer的router_logits形成的元组
                self.num_experts,
                self.num_experts_per_tok,
            ) 
            
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss
        
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return ((loss,) + output) if loss is not None else output
        
        
        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
        
        
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        # 如果有缓存，说明不是第一次生成，只需保留最后一个token
        if past_key_values:
            input_ids = input_ids[:, -1:]   # 保持二维形状 [batch_size, 1]
            
        # 处理位置编码
        position_ids = kwargs.get("position_ids", None)
        
        if attention_mask is not None and position_ids is None:
            # 动态生成位置ID：cumsum累计求和-1得到绝对位置
            # 例如 attention_mask = [1,1,1] → [0,1,2]
            position_ids = attention_mask.long().cumsum(-1)-1
            
            # 将padding位置设为1（避免无效位置影响）
            position_ids.masked_fill(attention_mask==0, 1)

            # 如果有缓存，只需使用最后一个位置
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1) # # [batch_size, 1]
                
                
        
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            # 只在第一次生成时使用embedding输入
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
           
        
        # 组装最终输入字典
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,  # 历史缓存
            "use_cache": kwargs.get("use_cache"),  # 是否使用缓存加速
            "attention_mask": attention_mask,  # 注意力掩码
        })
        
        return model_inputs