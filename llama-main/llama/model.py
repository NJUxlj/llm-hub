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

from typing import List, Optional, Tuple, Dict

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
    
        
    基础频率计算：
        theta=10000.0控制频率衰减速度
        生成维度索引：i ∈ [0, 2, 4,...,dim-2]
        计算公式：1/(theta^(i/dim)) → 产生按维度指数衰减的频率
    
    外积运算：
        将位置索引t（时间步）与基础频率相乘
        生成二维矩阵：freqs[t,i] = t * (theta^(-2i/dim)) # 相当于让每个 token 对应一个频率向量
     
    复数转换：
        通过torch.polar实现欧拉公式：e^(iθ) = cosθ + i sinθ
        每个元素对应复数：(cos(t*theta^(-2i/dim)), sin(t*theta^(-2i/dim)))
        
        
    ##Return
        返回一个预先计算好的包含复指数的频率张量。

        生成一个复数张量，数据类型为complex64
        每个元素的形式为：abs(cos(freqs) + i * sin(freqs)) = abs* e^{i*freqs}, abs是模长， 在本例中，模长为1
        这个复数张量将用于后续的旋转嵌入（Rotary Embedding）计算
        
        return freqs_cis: shape = [seq_len, dim // 2]
    '''
    
    # 这里使用了 torch.arange 生成从 0 到 dim 步长为 2 的张量，然后取前 dim // 2 个元素
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)) # shape = [dim // 2]
    # 序列位置索引
    t = torch.arange(end, device=freqs.device)  # type: ignore
    # # 计算外积，得到一个二维张量，形状为 (end, dim // 2)， 相当于让每个 token 对应一个频率向量
    freqs = torch.outer(t, freqs).float()  # type: ignore
    # 使用 torch.polar 函数将幅值为 1 的实部和 freqs 作为相位，生成复数指数张量 （欧拉公式实现） 
    # 这里的复数张量数据类型为 complex64
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64  # shape
    
    '''
    1. torch.ones_like(freqs):

        创建一个与freqs张量形状相同的全1张量
        这些1将作为复数的模（magnitude）
        
    2. freqs:

        这是之前计算得到的频率张量
        这些值将作为复数的相位（phase）
        
        
    3. torch.polar():

        这是PyTorch的极坐标转换函数
        根据给定的模和相位生成复数
        数学公式：torch.polar(abs, angle) = abs * (cos(angle) + i * sin(angle))
        这里相当于执行欧拉公式：e^(iθ) = cosθ + i sinθ
    
    4. 结果:

        生成一个复数张量，数据类型为complex64
        每个元素的形式为：abs(cos(freqs) + i * sin(freqs)), abs是模长， 在本例中，模长为1
        这个复数张量将用于后续的旋转嵌入（Rotary Embedding）计算

    5. 为什么需要这样做:

        在Transformer中，使用复数表示可以方便地实现旋转操作
        通过复数乘法，可以高效地实现位置信息的编码
        这种表示方法有助于模型更好地捕捉序列中的位置关系
        举例说明： 假设freqs中的一个元素是0.5，那么生成的复数将是： cos(0.5) + i * sin(0.5) ≈ 0.8776 + 0.4794i

    这种复数表示将在后续的apply_rotary_emb函数中用于对query和key进行旋转操作，从而为模型提供位置信息。
    '''
    return freqs_cis





def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped. shape = [seq_len, dim//2]
        x (torch.Tensor): Target tensor for broadcasting compatibility. x.shape = [batch_size, seq_len, dim] or [batch_size, seq_len, n_heads, head_dim//2, 2]

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
    freqs_cis: torch.Tensor, # 预先定义好的三角频率矩阵，矩阵中的每个元素是 abs(cos(theta)+i*sin(theta))，其中， theta = 1/(10000 ^ (2i / dim)),
) -> Tuple[torch.Tensor, torch.Tensor]:
    
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query 'xq' and key 'xk' tensors using the provided
    frequency tensor 'freqs_cis'. The input tensors are reshaped as complex numbers, and the frequency tensor
    is reshaped for broadcasting compatibility. The resulting tensors contain rotary embeddings and are
    returned as real tensors.
    
    

    Args:
        xq (torch.Tensor): Query tensor to apply rotary embeddings. shape= [batch_size, seq_len, n_heads, head_dim]
        xk (torch.Tensor): Key tensor to apply rotary embeddings. shape= [batch_size, seq_len, n_heads, head_dim]
        freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials. shape = [seq_len, dim//2]

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

        return xq, xk, where xq.shape == xk.shape [batch_size, seq_len, n_heads, head_dim]

    """
    # 首先将query和key张量重塑并转换为复数形式。这里通过`reshape(*xq.shape[:-1], -1, 2)`将最后一个维度每两个数字组合成一个复数，其中:
    # - 实部和虚部分别对应embedding维度中相邻的两个值
        # 每对相邻的实数 $(x, y)$ 被视为一个复数 $z = x + yi$
        # 这种表示便于后续进行复数乘法（旋转操作）
    # - `view_as_complex`将这些数对解释为复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # shape = [batch_size, seq_len, n_heads, head_dim//2, 2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    # freqs_cis.shape = [seq_len, dim//2]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # [1, seq_len, 1, head_dim//2, 1] 
    
    # 应用旋转
    # xq_.shape = [batch_size, seq_len, n_heads, head_dim//2, 2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)   # 对位相乘， 然后将复数转回实数对， 然后 flatten(3) 将最后两个维度合并
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    
    # - 通过复数乘法实现旋转变换
    # - `view_as_real`将结果转回实数域
    # - `flatten(3)`将最后两个维度压缩回原始形状 

    return xq_out.type_as(xq), xk_out.type_as(xk)
    
    
    


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )





class Attention(nn.Module):
    """Multi-head attention module."""
    def __init__(self, args: ModelArgs):
        """
        Initialize the Attention module.

        Args:
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_kv_heads (int): Number of key and value heads.
            n_local_heads (int): Number of local query heads.
            n_local_kv_heads (int): Number of local key and value heads.
            n_rep (int): Number of repetitions for local heads.
            head_dim (int): Dimension size of each attention head.
            wq (ColumnParallelLinear): Linear transformation for queries.
            wk (ColumnParallelLinear): Linear transformation for keys.
            wv (ColumnParallelLinear): Linear transformation for values.
            wo (RowParallelLinear): Linear transformation for output.
            cache_k (torch.Tensor): Cached keys for attention.
            cache_v (torch.Tensor): Cached values for attention.

        """
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 总共的GPU数量
        model_parallel_size = fs_init.get_model_parallel_world_size()
        # 每个GPU上的头数量
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 在 multi-query attention中， query的头数 = n_heads, kv的头数 <= n_heads
        # 因此，在做注意力计算的时候，我们需要将 kv的头数复制 n_rep 份，kv.n_heads = q.n_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        
        # 这是一个并行线性层，用于在模型并行设置中处理线性变换。
        # 它将输入张量按列分割到不同的设备上，以提高计算效率。
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False, # 表示不收集并行计算的结果。在模型并行设置中，输出会分布在不同的设备上，如果设置为False，则保持这种分布状态。
            init_method=lambda x: x, # 这是一个初始化方法，使用恒等函数（identity function）作为初始化器。
                                        # 这意味着权重矩阵会保持初始值不变，通常在实际使用中会使用其他初始化方法。
        )
        
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias = False,
            gather_output = False,
            init_method=lambda x: x,
        )
        
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias = False,
            gather_output = False,
            init_method=lambda x: x,
        )
        
        
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel = True,
            init_method = lambda x:x,
        )
        
        # cache_k, cache_v 相当于是模型之前推理文本保存的历史记录, 每forward推理一次，就把当前的input_embedding 保存到cache里
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
            ).cuda()
        
        
        self.cache_v = torch.zeros(
            (
             args.max_batch_size,
             args.max_seq_len,
             self.n_local_kv_heads,
             self.head_dim,
                
            )
            ).cuda()
        
        
    
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor], # padding 掩码
    ):
        """
        Forward pass of the attention module.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for caching.
            freqs_cis (torch.Tensor): Precomputed frequency tensor. [cosine, sine]
            mask (torch.Tensor, optional): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after attention.

        """
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        
        xq, xk = self.apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        
        self.cache_k[:bsz, start_pos: start_pos + seqlen]  = xk
        self.cache_v[:bsz, start_pos: start_pos + seqlen]  = xv
        
        keys = self.cache_k[:bsz, :start_pos + seqlen]
        values = self.cache_v[:bsz, :start_pos + seqlen]
        
        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, n_rep=self.n_rep) # (bs, seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, n_rep=self.n_rep) # (bs, seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2) # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores += mask # (bs, n_local_heads, seqlen, cache_len + seqlen) # where q_size = seqlen and k_size = cache_len + seqlen
        
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output) # shape = [batch_size, seq_len, dim]
        
        
    def apply_rotary_emb(
        self,
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
            xq (torch.Tensor): Query tensor to apply rotary embeddings. shape= [batch_size, seq_len, n_heads, head_dim]
            xk (torch.Tensor): Key tensor to apply rotary embeddings. shape= [batch_size, seq_len, n_heads, head_dim]
            freqs_cis (torch.Tensor): Precomputed frequency tensor for complex exponentials. shape = [seq_len, dim//2]

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.

            return xq, xk, where xq.shape == xk.shape [batch_size, seq_len, n_heads, head_dim]

        """
        
        
        # 首先将query和key张量重塑并转换为复数形式。这里通过`reshape(*xq.shape[:-1], -1, 2)`将最后一个维度每两个数字组合成一个复数，其中:
        # - 实部和虚部分别对应embedding维度中相邻的两个值
            # 每对相邻的实数 $(x, y)$ 被视为一个复数 $z = x + yi$
            # 这种表示便于后续进行复数乘法（旋转操作）
        # - `view_as_complex`将这些数对解释为复数
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) 
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # [1, seq_len, 1, head_dim//2, 1]
        
        # 应用旋转
        # xq_.shape = [batch_size, seq_len, n_heads, head_dim//2, 2]
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)   # 对位相乘， 然后将复数转回实数对， 然后 flatten(3) 将最后两个维度合并
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        
        # - 通过复数乘法实现旋转变换
        # - `view_as_real`将结果转回实数域
        # - `flatten(3)`将最后两个维度压缩回原始形状 

        return xq_out.type_as(xq), xk_out.type_as(xk)







class FeedForward(nn.Module):
    def __init__(
       self,
       dim: int,
       hidden_dim:int,
       multiple_of:int,
       ffn_dim_multiplier:Optional[float]
    ):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.
            ffn_dim_multiplier (float, optional): Custom multiplier for hidden dimension. Defaults to None.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
            
        hidden_dim  = multiple_of * ((hidden_dim + multiple_of -1) // multiple_of)
            
            
        # 模型并行：将权重矩阵按列拆分到不同设备，每个设备计算部分结果
        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        
        
        # 权重矩阵形状为 [hidden_dim, dim]
        # 按行拆分到不同设备，例如设备1持有 W2[0:hidden_dim//n, :]，设备2持有 W2[hidden_dim//n:, :]
        
         # - 输入x已经提前被多头分块好了 ：[bsz, n, b, dim/n]
        # 设备1计算：x[..., :dim/n] @ W2[:dim/n, hidden_dim].T → output_part1
        # 设备2计算：x[..., dim/n:] @ W2[dim/n:, hidden_dim].T → output_part2
        # 设备 n....
        # 最终输出 = output_part1 + output_part2 + output_partn
        
        # 这里涉及到分块矩阵乘法，可以在草稿纸上尝试：A1*B1 + A2*B2 = A*B [其中，A1,A2是x按列分块, B1, B2 是 W 按行分块]
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_parallel=True, init_method=lambda x: x
        )
        
        self.w3 = ColumnParallelLinear( # gather_output=False: 每个设备上单独做运算，不需要聚集所有GPU的结果
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
            
        )
        
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
    
    
    


class TransformerBlock(nn.Module):
    def __init__(self, layer_id:int, args:ModelArgs):
        """
        Initialize a TransformerBlock.

        Args:
            layer_id (int): Identifier for the layer.
            args (ModelArgs): Model configuration parameters.

        Attributes:
            n_heads (int): Number of attention heads.
            dim (int): Dimension size of the model.
            head_dim (int): Dimension size of each attention head.
            attention (Attention): Attention module.
            feed_forward (FeedForward): FeedForward module.
            layer_id (int): Identifier for the layer.
            attention_norm (RMSNorm): Layer normalization for attention output.
            ffn_norm (RMSNorm): Layer normalization for feedforward output.

        """
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        
        self.attention= Attention(args)
        
        self.feed_forward = FeedForward(
            dim = args.dim,
            hidden_dim = 4*args.dim,
            multiple_of = args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        
        self.layer_id = layer_id
        
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        
        
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor]
        ):
        """
        Perform a forward pass through the TransformerBlock.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position for attention caching.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.
            mask (torch.Tensor, optional): Masking tensor for attention. Defaults to None.

        Returns:
            torch.Tensor: Output tensor after applying attention and feedforward layers.

        """
        # 比起 post-norm, pre-norm 的训练效果更稳定，但是效果略差
        h = x + self.attention.forward(
            self.attention_norm(x), start_pos, freqs_cis, mask
        )

        out = h + self.feed_forward.forward(self.ffn_norm(h))
        
        return out



class Transformer(nn.Module):
    
    
    def __init__(self, params:ModelArgs):
        """
        Initialize a Transformer model.

        Args:
            params (ModelArgs): Model configuration parameters.

        Attributes:
            params (ModelArgs): Model configuration parameters.
            vocab_size (int): Vocabulary size.
            n_layers (int): Number of layers in the model.
            tok_embeddings (ParallelEmbedding): Token embeddings.
            layers (torch.nn.ModuleList): List of Transformer blocks.
            norm (RMSNorm): Layer normalization for the model output.
            output (ColumnParallelLinear): Linear layer for final output.
            freqs_cis (torch.Tensor): Precomputed cosine and sine frequencies.

        """
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers
        
        
        
        self.tok_embeddings = ParallelEmbedding(
            params.vocab_size, params.dim, init_method = lambda x:x
        )
        
        
        self.layers:List[TransformerBlock] = nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(   # LM head
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )
        
        
        self.freqs_cis = precompute_freqs_cis(
            # Note that self.params.max_seq_len is multiplied by 2 because the token limit for the Llama 2 generation of models is 4096. 
            # Adding this multiplier instead of using 4096 directly allows for dynamism of token lengths while training or fine-tuning.
            self.params.dim // self.params.n_heads, self.params.max_seq_len * 2
        )
        
        
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        
        mask = None
        if seqlen > 1:
            mask = torch.full(
                (1, 1, seqlen, seqlen), float("-inf"), device = tokens.device
            )
            # 创建上三角矩阵，从主对角线偏移 (start_pos+1) 的位置开始
            # diagonal=start_pos+1实现动态偏移：
                # 初始生成时（start_pos=0）：对角线偏移1，形成严格上三角
                # 增量生成时（start_pos>0）：保留历史注意力的同时屏蔽未来token
            mask = torch.triu(mask, diagonal =  start_pos +1 ).type_as(h) 
            # start_pos 及其之前，全是历史记录， 是Transformer模型中，最初始的cache
            # 换个角度理解， 对角线上移其实是把 attention mask 左侧的前 start_pos 列全部宅出来， 作为历史记录 ，剩余的 seqlen-start_pos 列，可以看做一个常规的attention矩阵。
        for layer in self.layers:
            h = layer.forward(h, start_pos, freqs_cis = self.freqs_cis, mask = mask)
        h = self.norm(h)
        
        output = self.output(h).float()
        return output