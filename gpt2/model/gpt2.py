import math
import os
import logging
import json
import torch
import torch.nn as nn
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass
from torch.nn import functional as F
from embeddings import GPT2Embeddings
from config.model_config import GPT2Config

from attention import CausalSelfAttention

from pathlib import Path


@dataclass
class GPT2Output:
    """
    GPT2模型输出的数据类
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None




class GPT2MLP(nn.Module):
    def init(self, config: GPT2Config):
        """
        GPT2 MLP层

            Args:  
                config: 模型配置对象  
        """  
        super().__init__()  
        self.c_fc = nn.Linear(config.hidden_size, config.intermediate_size)  
        self.c_proj = nn.Linear(config.intermediate_size, config.hidden_size)  
        self.dropout = nn.Dropout(config.resid_dropout)  
        self.activation = F.gelu  

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:  
        hidden_states = self.c_fc(hidden_states)  
        hidden_states = self.activation(hidden_states)  
        hidden_states = self.c_proj(hidden_states)  
        hidden_states = self.dropout(hidden_states)  
        return hidden_states 



# class GPT2Block(nn.Module):
#     """GPT-2 Transformer块"""
#     def __init__(self, config):
#         super().__init__()
#         self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
#         self.attn = CausalSelfAttention(config)
#         self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
#         self.mlp = nn.Sequential(
#             nn.Linear(config.n_embd, config.n_inner),
#             nn.GELU(),
#             nn.Linear(config.n_inner, config.n_embd),
#             nn.Dropout(config.resid_pdrop),
#         )

#     def forward(self, x):
#         x = x + self.attn(self.ln_1(x))
#         x = x + self.mlp(self.ln_2(x))
#         return x
    


class GPT2Block(nn.Module):
    def init(self, config: GPT2Config):
        """
        GPT2 Transformer块

        Args:  
            config: 模型配置对象  
        """  
        super().__init__()  
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)  
        self.attn = CausalSelfAttention(config)  
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)  
        self.mlp = GPT2MLP(config)  

    def forward(  
        self,  
        hidden_states: torch.Tensor,  
        attention_mask: Optional[torch.Tensor] = None,  
        layer_past: Optional[Tuple[torch.Tensor]] = None,  
        use_cache: bool = False,  
        output_attentions: bool = False,  
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, ...]]]:  
        residual = hidden_states  
        hidden_states = self.ln_1(hidden_states)  
        
        # Self Attention  
        attn_outputs = self.attn(  
            hidden_states,  
            attention_mask=attention_mask,  
            layer_past=layer_past,  
            use_cache=use_cache,  
            output_attentions=output_attentions,  
        )  
        
        # 处理注意力输出  
        attn_output = attn_outputs[0]  # 注意力输出  
        outputs = attn_outputs[1:]     # 过去层的缓存和注意力权重（如果需要）  
        
        # 残差连接  
        hidden_states = attn_output + residual  
        
        # MLP  
        residual = hidden_states  
        hidden_states = self.ln_2(hidden_states)  
        hidden_states = self.mlp(hidden_states)  
        hidden_states = hidden_states + residual  

        if use_cache:  
            outputs = (hidden_states,) + outputs  
        else:  
            outputs = (hidden_states,) + outputs[1:]  

        return outputs  



# class GPT2(nn.Module):
#     """GPT-2模型实现"""
#     def __init__(self, config):
#         super().__init__()
#         self.config = config
        
#         # input embedding
#         self.wte = nn.Embedding(config.vocab_size, config.n_embd)
#         self.wpe = nn.Embedding(config.n_positions, config.n_embd)
#         self.drop = nn.Dropout(config.embd_pdrop)
        
#         # transformer blocks
#         self.blocks = nn.ModuleList([GPT2Block(config) for _ in range(config.n_layer)])
        
#         # final layer norm
#         self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
#         # initialize weights
#         self.apply(self._init_weights)

#     def _init_weights(self, module):
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#             if isinstance(module, nn.Linear) and module.bias is not None:
#                 module.bias.data.zero_()
#         elif isinstance(module, nn.LayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)

#     def forward(self, input_ids, position_ids=None):
#         b, t = input_ids.size()
        
#         # 生成位置编码
#         if position_ids is None:
#             position_ids = torch.arange(0, t, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        
#         # 前向传播
#         token_embeddings = self.wte(input_ids)
#         position_embeddings = self.wpe(position_ids)
#         x = self.drop(token_embeddings + position_embeddings)
        
#         for block in self.blocks:
#             x = block(x)
        
#         x = self.ln_f(x)
        
#         # 输出logits
#         logits = F.linear(x, self.wte.weight)
        
#         return logits

#     def generate(self, input_ids, max_length, temperature=1.0, top_k=None):
#         """简单的文本生成函数"""
#         self.eval()
#         with torch.no_grad():
#             for _ in range(max_length - input_ids.size(1)):
#                 # 获取预测
#                 logits = self(input_ids)
#                 logits = logits[:, -1, :] / temperature
                
#                 # top-k采样
#                 if top_k is not None:
#                     v, _ = torch.topk(logits, top_k)
#                     logits[logits < v[:, [-1]]] = float('-inf')
                
#                 # 采样下一个token
#                 probs = F.softmax(logits, dim=-1)
#                 next_token = torch.multinomial(probs, num_samples=1)
                
#                 # 添加到输入序列
#                 input_ids = torch.cat([input_ids, next_token], dim=1)
                
#         return input_ids
    
    
    


class GPT2Model(nn.Module):
    def init(self, config: GPT2Config):
        """
        GPT2 基础模型

        Args:  
            config: 模型配置对象  
        """  
        super().__init__()  
        self.config = config  
        
        self.embeddings = GPT2Embeddings(config)  
        self.dropout = nn.Dropout(config.embd_dropout)  
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_hidden_layers)])  
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)  

def forward(  
    self,  
    input_ids: torch.LongTensor,  
    attention_mask: Optional[torch.Tensor] = None,  
    position_ids: Optional[torch.LongTensor] = None,  
    past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  
    use_cache: Optional[bool] = None,  
    output_attentions: Optional[bool] = None,  
    output_hidden_states: Optional[bool] = None,  
) -> Union[Tuple[torch.Tensor], GPT2Output]:  
    
    # 初始化输出参数  
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions  
    output_hidden_states = (  
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states  
    )  
    use_cache = use_cache if use_cache is not None else self.config.use_cache  

    # 获取输入嵌入  
    hidden_states = self.embeddings(input_ids, position_ids=position_ids)  
    hidden_states = self.dropout(hidden_states)  

    # 初始化注意力掩码  
    if attention_mask is not None:  
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))  
        attention_mask = attention_mask[:, None, None, :]  
        attention_mask = attention_mask.to(dtype=hidden_states.dtype)  
        attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min  

    # 初始化输出列表  
    all_hidden_states = () if output_hidden_states else None  
    all_attentions = () if output_attentions else None  
    next_decoder_cache = () if use_cache else None  

    # 通过所有transformer层  
    for i, block in enumerate(self.h):  
        if output_hidden_states:  
            all_hidden_states = all_hidden_states + (hidden_states,)  

        layer_past = past_key_values[i] if past_key_values is not None else None  

        layer_outputs = block(  
            hidden_states,  
            attention_mask=attention_mask,  
            layer_past=layer_past,  
            use_cache=use_cache,  
            output_attentions=output_attentions,  
        )  

        hidden_states = layer_outputs[0]  

        if use_cache:  
            next_decoder_cache += (layer_outputs[1],)  

        if output_attentions:  
            all_attentions = all_attentions + (layer_outputs[2 if use_cache else 1],)  

    # 最后的层归一化  
    hidden_states = self.ln_f(hidden_states)  

    if output_hidden_states:  
        all_hidden_states = all_hidden_states + (hidden_states,)  

    return GPT2Output(  
        last_hidden_state=hidden_states,  
        past_key_values=next_decoder_cache,  
        hidden_states=all_hidden_states,  
        attentions=all_attentions,  
    )  
    
    
    
    




class GPT2LMHeadModel(nn.Module):
    def init(self, config: GPT2Config):
        """
        GPT2 语言模型（带有语言模型头）

        Args:  
            config: 模型配置对象  
        """  
        super().__init__()  
        self.config = config  
        self.transformer = GPT2Model(config)  
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  

        # 初始化权重  
        self.apply(self._init_weights)  
        
        # 将语言模型头的权重与词嵌入绑定  
        self.tie_weights()  

    def _init_weights(self, module):  
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

    def tie_weights(self):  
        """  
        将输出层的权重与输入嵌入层的权重绑定  
        """  
        self.lm_head.weight = self.transformer.embeddings.word_embeddings.weight  

    def forward(  
        self,  
        input_ids: torch.LongTensor,  
        attention_mask: Optional[torch.Tensor] = None,  
        position_ids: Optional[torch.LongTensor] = None,  
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  
        labels: Optional[torch.LongTensor] = None,  
        use_cache: Optional[bool] = None,  
        output_attentions: Optional[bool] = None,  
        output_hidden_states: Optional[bool] = None,  
    ) -> GPT2Output:  
        """  
        前向传播  
        
        Args:  
            input_ids: 输入序列  
            attention_mask: 注意力掩码  
            position_ids: 位置编码  
            past_key_values: 过去的键值对缓存  
            labels: 目标标签  
            use_cache: 是否使用过去的键值对缓存  
            output_attentions: 是否输出注意力权重  
            output_hidden_states: 是否输出隐藏状态  
        """  
        transformer_outputs = self.transformer(  
            input_ids,  
            attention_mask=attention_mask,  
            position_ids=position_ids,  
            past_key_values=past_key_values,  
            use_cache=use_cache,  
            output_attentions=output_attentions,  
            output_hidden_states=output_hidden_states,  
        )  

        hidden_states = transformer_outputs.last_hidden_state  
        logits = self.lm_head(hidden_states)  

        loss = None  
        if labels is not None:  
            # 将logits和labels展平  
            shift_logits = logits[..., :-1, :].contiguous()  
            shift_labels = labels[..., 1:].contiguous()  
            
            # 计算损失  
            loss_fct = nn.CrossEntropyLoss()  
            loss = loss_fct(  
                shift_logits.view(-1, shift_logits.size(-1)),  
                shift_labels.view(-1)  
            )  

        return GPT2Output(  
            loss=loss,  
            logits=logits,  
            hidden_states=transformer_outputs.hidden_states,  
            attentions=transformer_outputs.attentions,  
        )  

    def generate(  
        self,  
        input_ids: torch.LongTensor,  
        max_length: int,  
        temperature: float = 1.0,  
        do_sample: bool = True,  
        top_k: int = 50,  
        top_p: float = 0.95,  
        repetition_penalty: float = 1.0,  
        num_return_sequences: int = 1,  
        **kwargs  
    ) -> torch.LongTensor:  
        """  
        生成文本  
        
        Args:  
            input_ids: 输入序列  
            max_length: 最大生成长度  
            temperature: 温度参数  
            do_sample: 是否使用采样  
            top_k: top-k采样参数  
            top_p: nucleus采样参数  
            repetition_penalty: 重复惩罚参数  
            num_return_sequences: 返回序列数量  
        """  
        # 设置为评估模式  
        self.eval()  
        
        # 初始化生成参数  
        batch_size = input_ids.shape[0]  
        cur_len = input_ids.shape[1]  
        
        # 确保输入正确的设备  
        device = input_ids.device  
        
        # 初始化输出序列  
        output_sequences = input_ids.clone()  
        
        with torch.no_grad():  
            for _ in range(max_length - cur_len):  
                # 获取模型输出  
                outputs = self(output_sequences, use_cache=True)  
                next_token_logits = outputs.logits[:, -1, :]  
                
                # 应用温度  
                next_token_logits = next_token_logits / temperature  
                
                # 应用重复惩罚  
                if repetition_penalty != 1.0:  
                    for i in range(batch_size):  
                        for previous_token in set(output_sequences[i].tolist()):  
                            next_token_logits[i, previous_token] /= repetition_penalty  
                
                # Top-K采样  
                if top_k > 0:  
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]  
                    next_token_logits[indices_to_remove] = float('-inf')  
                
                # Top-p采样  
                if top_p < 1.0:  
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)  
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)  
                    
                    sorted_indices_to_remove = cumulative_probs > top_p  
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()  
                    sorted_indices_to_remove[..., 0] = 0  
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)  
                    next_token_logits[indices_to_remove] = float('-inf')  
                
                # 采样或贪婪解码  
                if do_sample:  
                    probs = F.softmax(next_token_logits, dim=-1)  
                    next_tokens = torch.multinomial(probs, num_samples=1)  
                else:  
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  
                
                # 将新token添加到序列中  
                output_sequences = torch.cat([output_sequences, next_tokens], dim=-1)  
                
                # 检查是否生成了结束符号  
                if (next_tokens == self.config.eos_token_id).any():  
                    break  
        
        return output_sequences  
    
    def save_pretrained(  
        self,  
        save_directory: Union[str, Path],  
        save_config: bool = True,  
        state_dict: Optional[Dict[str, Any]] = None,  
        save_function: callable = torch.save,  
    ) -> None:  
        """  
        保存模型权重和配置到指定目录  
        
        Args:  
            save_directory: 保存模型的目录路径  
            save_config: 是否同时保存配置文件  
            state_dict: 可选的状态字典，如果为None则使用当前模型的状态  
            save_function: 保存模型的函数，默认为torch.save  
            
        Raises:  
            AssertionError: 如果保存目录已存在且非空  
        """  
        save_directory = Path(save_directory)  
        os.makedirs(save_directory, exist_ok=True)  
        
        # 确保目录为空或不存在  
        if len(os.listdir(save_directory)) > 0:  
            logging.warning(f"目录 {save_directory} 非空，文件可能被覆盖")  
        
        # 保存模型权重  
        if state_dict is None:  
            state_dict = self.state_dict()  
            
        # 保存模型权重  
        model_file = save_directory / "pytorch_model.bin"  
        save_function(state_dict, model_file)  
        logging.info(f"模型权重已保存至 {model_file}")  
        
        # 保存配置文件  
        if save_config:  
            config_file = save_directory / "config.json"  
            with open(config_file, 'w', encoding='utf-8') as f:  
                # 将配置对象转换为字典  
                config_dict = {  
                    "model_type": "gpt2",  
                    "n_positions": self.config.n_positions,  
                    "n_embd": self.config.n_embd,  
                    "n_layer": self.config.n_layer,  
                    "n_head": self.config.n_head,  
                    "activation_function": self.config.activation_function,  
                    "resid_pdrop": self.config.resid_pdrop,  
                    "embd_pdrop": self.config.embd_pdrop,  
                    "attn_pdrop": self.config.attn_pdrop,  
                    "layer_norm_epsilon": self.config.layer_norm_epsilon,  
                    "initializer_range": self.config.initializer_range,  
                    "vocab_size": self.config.vocab_size,  
                    # 添加其他配置参数...  
                }  
                json.dump(config_dict, f, indent=2)  
            logging.info(f"模型配置已保存至 {config_file}")  
            
        # 保存特殊文件  
        # 创建一个README文件说明模型信息  
        readme_content =  f"""# GPT-2 Model  
            This model is a custom implementation of GPT-2 saved at {save_directory}.

            Model Details
            Layers: {self.config.n_layer}
            Hidden Size: {self.config.n_embd}
            Attention Heads: {self.config.n_head}
            Position Embeddings: {self.config.n_positions}
            Vocabulary Size: {self.config.vocab_size}
            Usage
            from model.gpt2 import GPT2LMHeadModel  
            from config.model_config import ModelConfig  

            # Load configuration  
            config = ModelConfig.from_json_file('config.json')  

            # Initialize model  
            model = GPT2LMHeadModel(config)  

            # Load weights  
            state_dict = torch.load('pytorch_model.bin')  
            model.load_state_dict(state_dict)  
            """
        with open(save_directory / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme_content)
            
            
    @classmethod  
    def from_pretrained(  
        cls,  
        pretrained_model_path: Union[str, Path],  
        config: Optional[GPT2Config] = None,  
        map_location: Optional[Union[str, torch.device]] = None  
    ) -> "GPT2LMHeadModel":  
        """  
        从预训练文件加载模型  
        
        Args:  
            pretrained_model_path: 预训练模型目录路径  
            config: 可选的模型配置，如果为None则从目录中加载  
            map_location: 模型加载位置（CPU/GPU）  
            
        Returns:  
            加载了预训练权重的模型实例  
            
        Raises:  
            FileNotFoundError: 如果模型文件或配置文件不存在  
        """  
        pretrained_model_path = Path(pretrained_model_path)  
        
        # 加载配置  
        if config is None:  
            config_file = pretrained_model_path / "config.json"  
            if not config_file.exists():  
                raise FileNotFoundError(f"在 {pretrained_model_path} 中未找到config.json")  
                
            with open(config_file, 'r', encoding='utf-8') as f:  
                config_dict = json.load(f)  
                config = GPT2Config(**config_dict)  
        
        # 初始化模型  
        model = cls(config)  
        
        # 加载模型权重  
        model_file = pretrained_model_path / "pytorch_model.bin"  
        if not model_file.exists():  
            raise FileNotFoundError(f"在 {pretrained_model_path} 中未找到pytorch_model.bin")  
            
        state_dict = torch.load(model_file, map_location=map_location)  
        model.load_state_dict(state_dict)  
        
        logging.info(f"模型已从 {pretrained_model_path} 加载")  
        return model  
                
                
                




            
            


