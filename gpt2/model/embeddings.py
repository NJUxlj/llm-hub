import math
import torch
import torch.nn as nn

class GPT2Embeddings(nn.Module):
    """GPT2的嵌入层，包含词嵌入、位置嵌入和层标准化"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 词嵌入层
        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            config.n_embd,
            padding_idx=config.pad_token_id
        )
        
        # 位置嵌入层
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings,
            config.n_embd
        )
        
        # Layer Norm层
        self.LayerNorm = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.embd_pdrop)
        
        # 位置编码缓存
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )
        
    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
            
        seq_length = inputs_embeds.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            
        # 获取位置嵌入
        position_embeds = self.position_embeddings(position_ids)
        
        # 合并词嵌入和位置嵌入
        embeddings = inputs_embeds + position_embeds
        
        # 应用层标准化和dropout
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings