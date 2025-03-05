import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from enum import Enum

import math

def gelu_new(x):
    """
    Implementation of the GELU activation function.
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def create_attention_mask(input_ids: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    创建注意力掩码
    Args:
        input_ids: 输入序列 [batch_size, seq_length]
        padding_mask: 填充掩码 [batch_size, seq_length]
    Returns:
        attention_mask: [batch_size, 1, seq_length, seq_length]
    """
    batch_size, seq_length = input_ids.shape
    
    # 创建因果掩码（上三角矩阵）
    causal_mask = torch.triu(
        torch.ones((seq_length, seq_length), dtype=torch.bool), diagonal=1
    )
    
    # 扩展维度以适应batch_size
    causal_mask = causal_mask.expand(batch_size, 1, seq_length, seq_length)
    
    if padding_mask is not None:
        # 将padding_mask转换为注意力掩码格式
        padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        padding_mask = padding_mask.expand(batch_size, 1, seq_length, seq_length)
        causal_mask = causal_mask | padding_mask
        
    return causal_mask

def init_weights(module):
    """
    初始化模型权重
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def get_model_param_count(model: nn.Module) -> Tuple[int, int]:
    """
    获取模型参数数量
    Returns:
        total_params: 总参数量
        trainable_params: 可训练参数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params








def add_end_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = (fn.__doc__ if fn.__doc__ is not None else "") + "".join(docstr)
        return fn

    return docstring_decorator







class ExplicitEnum(str, Enum):
    """
    Enum with more explicit error message for missing values.
    """

    @classmethod
    def _missing_(cls, value):
        raise ValueError(
            f"{value} is not a valid {cls.__name__}, please select one of {list(cls._value2member_map_.keys())}"
        )



class TensorType(ExplicitEnum):
    """
    Possible values for the `return_tensors` argument in [`PreTrainedTokenizerBase.__call__`]. Useful for
    tab-completion in an IDE.
    """

    PYTORCH = "pt"
    TENSORFLOW = "tf"
    NUMPY = "np"
    JAX = "jax"
    MLX = "mlx"

