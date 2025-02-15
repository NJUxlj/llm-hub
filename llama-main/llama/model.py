# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn



@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    # 这个参数的作用是确保 SwiGLU 激活函数所在的前馈FFN的隐藏层大小是 2 的较大幂次的倍数。
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None # 动态地调整隐藏层的维度大小
    norm_eps: float = 1e-5

    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x)->torch.Tensor:
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor. shape = [batch_size, seq_len, hidden_size]

        Returns:
            torch.Tensor: The normalized tensor.

        """
        
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    
    def forward(self, x)->torch.Tensor:
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
    
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
    
    
    
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    '''
    函数功能
        这个函数的主要作用是预先计算一个包含复指数（cis）的频率张量。复指数在数学上可以表示为 $e^{i\theta}$，在这个函数里，我们会根据给定的维度和结束索引来生成这样的频率张量。最终返回的张量包含的数据类型是 complex64，也就是复数类型。

    参数说明
        dim（整数）：频率张量的维度大小。可以把它想象成一个多维数组的某一个维度的长度。
        end（整数）：预先计算频率的结束索引。这个索引决定了我们要计算多少个频率值。
        theta（浮点数，可选）：频率计算的缩放因子，默认值是 10000.0。这个因子会影响频率的计算结果，就像一个调整频率大小的“旋钮”。
    返回值
        返回一个预先计算好的包含复指数的频率张量，数据类型为 torch.Tensor。这个张量可以在后续的计算中使用，比如在一些深度学习模型里用于旋转嵌入（rotary embeddings）的计算。

        return freqs_cis: shape = [seq_len, dim // 2]
        
    基础频率计算：
        theta=10000.0控制频率衰减速度
        生成维度索引：i ∈ [0, 2, 4,...,dim-2]
        计算公式：1/(theta^(i/dim)) → 产生按维度指数衰减的频率
    
    外积运算：
        将位置索引t（时间步）与基础频率相乘
        生成二维矩阵：freqs[t,i] = t * (theta^(-2i/dim))
    
    复数转换：
        通过torch.polar实现欧拉公式：e^(iθ) = cosθ + i sinθ
        每个元素对应复数：(cos(t*theta^(-2i/dim)), sin(t*theta^(-2i/dim)))
    '''
    
    # 这里使用了 torch.arange 生成从 0 到 dim 步长为 2 的张量，然后取前 dim // 2 个元素
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # shape = [dim // 2]
    # 序列位置索引
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # # 计算外积，得到一个二维张量，形状为 (end, dim // 2)
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # 使用 torch.polar 函数将幅值为 1 的实部和 freqs 作为相位，生成复数指数张量 （欧拉公式实现） 
    # 这里的复数张量数据类型为 complex64
    # math \text{out} = a*e^{j \theta} = abs*(cosθ + jsinθ) = \text{abs} \cdot \cos(\text{angle}) + \text{abs} \cdot \sin(\text{angle}) \cdot j
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64  # shape
    return freqs_cis





def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped. shape = [seq_len, dim//2]
        x (torch.Tensor): Target tensor for broadcasting compatibility. x.shape = [batch_size, seq_len, dim]

    Returns:
        torch.Tensor: Reshaped frequency tensor.
        
        return freqs_cis # shape = [1, seq_len, dim // 2] 

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    
    ndim = x.ndim # 获取目标张量 x 的维度数量：
    assert 0 <= 1 < ndim # 进行维度检查，确保目标张量 x 的维度数量至少为 2
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]) # shape = [seq_len, dim]
    
    # 对于目标张量 x 的每个维度，如果该维度的索引是 1 或者是最后一个维度，则保留该维度的大小；否则，将该维度的大小设置为 1
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape) # shape = [1, seq_len, dim // 2] 



    
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings.
        xk (torch.Tensor): Key tensor to apply rotary embeddings.
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

    """
