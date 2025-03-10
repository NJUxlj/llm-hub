# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Literal, Optional, Tuple, TypedDict

import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

from llama.model import ModelArgs, Transformer
from llama.tokenizer import Tokenizer

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str
    
    
class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: List[str]  # not required
    logprobs: List[float]  # not required
    
    


class ChatPrediction(TypedDict, total=False):
    generation: Message
    tokens: List[str]  # not required
    logprobs: List[float]  # not required


Dialog = List[Message]







class Llama:
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    )->"Llama":
        """
        Build a Llama instance by initializing and loading a pre-trained model.

        Args:
            ckpt_dir (str): Path to the directory containing checkpoint files.
            tokenizer_path (str): Path to the tokenizer file.
            max_seq_len (int): Maximum sequence length for input text.
            max_batch_size (int): Maximum batch size for inference.
            model_parallel_size (Optional[int], optional): Number of model parallel processes.
                If not provided, it's determined from the environment. Defaults to None.

        Returns:
            Llama: An instance of the Llama class with the loaded model and tokenizer.

        Raises:
            AssertionError: If there are no checkpoint files in the specified directory,
                or if the model parallel size does not match the number of checkpoint files.

        Note:
            This method initializes the distributed process group, sets the device to CUDA,
            and loads the pre-trained model and tokenizer.

        """
    
    
    
    
    def __init__(self):
        pass
    
    @torch.inference_mode
    def generate(
        self,
    ):
        pass
    
    
    
    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        pass
    
    
    
    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        pass
    
    
    




































def sample_top_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.

    """
    
    '''
    
    # 实现原理：
    1. 输入：已经降序排列的概率张量 probs_sort（形状 [batch_size, vocab_size]）
    2. 操作：沿最后一个维度（词汇表维度）计算累积和
    3. 输出：每个位置包含从最高概率到当前概率的总和

    # 示例说明：
    假设 probs_sort = [0.4, 0.3, 0.2, 0.1]
    经过 cumsum 后得到 probs_sum = [0.4, 0.7, 0.9, 1.0]

    # 后续配合 top-p 采样的应用：
    该累积和用于找出最小的索引 k，使得 sum(probs_sort[:k]) >= p
    从而确定需要保留的 token 范围，实现 nucleus sampling 的核心逻辑

    # 可视化过程：
    原始概率分布: ▁▃▅▇
    排序后分布: ▇▅▃▁
    累积和曲线: ▁▃▆█（当 p=0.9 时，会截断到第三个token）
    
    '''
    
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending = True)
    
    probs_sum = torch.cumsum(probs_sort, dim=-1) # 获取累计和数组
    
    
    '''
    应用示例： 当生成文本时，给定候选token的概率分布为[0.4, 0.3, 0.2, 0.1]，设置p=0.9时：

        保留前3个token（累计0.9）
        新概率分布为[0.44, 0.33, 0.22, 0]
        采样时可能选择到前3个token中的任意一个，但不会选到最后一个token
    '''
    mask = probs_sum - probs_sort > p
    # 将超出阈值范围的token概率置零
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True)) # 归一化
    next_token = torch.multinomial(probs_sort, num_samples=1) # 采样得到排序后的索引
    # 恢复原始索引：将采样结果映射回原始token顺序
    next_token = torch.gather(probs_idx, -1, next_token) # 以next_token 作为下标去收集probs_idx这个数组中的id
    return next_token



    '''
    案例：
    
    def sample_top_p(probs, p=0.8):
        # 步骤1：降序排序概率和索引
        probs_sort, probs_idx = torch.sort(torch.tensor([0.2, 0.7, 0.1]), 
                                        descending=True)  
        # probs_sort = [0.7, 0.2, 0.1], probs_idx = [1, 0, 2]

        # 步骤2：计算累积概率
        probs_sum = torch.cumsum(probs_sort, dim=-1)  # [0.7, 0.9, 1.0]

        # 步骤3：创建掩码（核心计算）
        mask = probs_sum - probs_sort > p
        # 计算过程：
        # 0.7 - 0.7 = 0.0 → False
        # 0.9 - 0.2 = 0.7 → False
        # 1.0 - 0.1 = 0.9 → True
        # mask = [False, False, True]

        # 步骤4：应用掩码
        probs_sort[mask] = 0.0  # 结果：[0.7, 0.2, 0.0]

        # 步骤5：重新归一化
        probs_sort /= probs_sort.sum()  # 0.7 + 0.2 = 0.9 → [0.7/0.9≈0.7778, 0.2/0.9≈0.2222, 0]

        # 步骤6：采样
        next_token = torch.multinomial(probs_sort, num_samples=1)  
        # 77.78%概率采样索引0（对应原始索引1），22.22%概率采样索引1（对应原始索引0）

        # 步骤7：恢复原始索引
        next_token = probs_idx[next_token]  
        # 可能返回 1（原始概率0.7）或 0（原始概率0.2）
    
    
    '''
    
    
    
    
    
    