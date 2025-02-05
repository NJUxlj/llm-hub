from torch import nn







class GLMBlock(nn.Module):
    def __init__(self, config: ChatGLMConfig, layer_number, device=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        