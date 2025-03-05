
from dataclasses import dataclass

@dataclass
class GPT2Config:
    """GPT-2模型配置类"""
    vocab_size: int = 50257  # GPT-2词表大小
    n_positions: int = 1024  # 最大序列长度
    n_embd: int = 768       # 嵌入维度
    n_layer: int = 12       # Transformer层数
    n_head: int = 12        # 注意力头数
    n_inner: int = None     # FFN内层维度，默认为4*n_embd
    activation_function: str = "gelu"  # 激活函数
    resid_pdrop: float = 0.1          # 残差dropout
    embd_pdrop: float = 0.1           # 嵌入dropout
    attn_pdrop: float = 0.1           # 注意力dropout
    layer_norm_epsilon: float = 1e-5   # Layer Norm epsilon
    initializer_range: float = 0.02    # 初始化范围
    scale_attn_weights: bool = True    # 是否缩放注意力权重
    use_cache: bool = True             # 是否使用past key/values缓存
    
    def __post_init__(self):
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd

