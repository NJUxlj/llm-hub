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


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."




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
    
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            if model_parallel_size is None:
                model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        # seed must be the same in all processes
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w") # 将标准输出重定向到 devnull, devnull是一个特殊的设备文件，写入其中的数据会被丢弃
            # 对于所有非主进程（local_rank > 0），它们的标准输出将被静默，只有主进程（local_rank = 0）会保留正常的输出功能

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}" # 确保至少找到一个检查点文件
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()] # 根据当前进程的并行rank选择对应的检查点文件
        checkpoint = torch.load(ckpt_path, map_location="cpu") # 加载检查点文件到CPU
        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor) # float16
        model = Transformer(model_args)
        '''
        strict=False 参数的含义：

            允许模型加载不完整的权重（比如checkpoint中缺少某些层的权重）
            但并不意味着只加载部分层
            如果checkpoint中缺少某些层的权重，这些层会保持初始化状态
        '''
        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Llama(model, tokenizer)
        
        
    
    
    
    def __init__(self, model:Transformer, tokenizer:Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    @torch.inference_mode
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ):
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len
        
        
    
    
    
    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
    
    
    
        if max_gen_len is None:
                max_gen_len = self.model.params.max_seq_len - 1

        
        prompt_tokens:List[List[int]] = [self.tokenizer.encode(x) for x in prompts]
        
        
        generation_tokens, generation_logprobs= self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        
        if logprobs:
            return [
                {
                    "generation": self.tokenizer.decode(t),
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i, # logprobs_i 是一个列表，包含了每个生成的 token 的 logprob
                    
                }
                for t, logprobs_i in zip(generation_tokens, generation_logprobs)
            ]
        else:
            return [{"generation":self.tokenizer.decode(t)} for t in generation_tokens]
    
    
    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens:List[List[int]] = []
        unsafe_requests = [] # shape = [batch_size]
        
        for dialog in dialogs:
            
            unsafe_requests.append(
                any([tag in msg['content'] for tag in SPECIAL_TAGS for msg in dialog]) 
            )
              
            if dialog[-1]["role"]== "system":    #  如果第一条是系统消息，将其与下一条用户消息合并
                dialog = [
                    {
                         "role":dialog[1]['role'],   # user
                         "content": B_SYS  # system begin token
                            +  dialog[0]["content"]
                            + E_SYS
                            +dialog[1]['content'],
                    },
                ]+ dialog[2:] # 保留对话中剩余的消息
                
        
            assert all([msg['role']=='user' for msg in dialog[::2]]) and all(
                [msg['role']=='system' for msg in dialog[1::2]]
            ),(
                "model only supports 'system', 'user' and 'assistant' roles, "
                    "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )

        
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {prompt['content'].strip()} {E_INST} {answer['content'].strip()}",
                        bos = True,
                        eos = True,
                    )
                    for prompt ,answer in zip(
                        dialog[::2], # user
                        dialog[1::2]
                    )
                ],
                []
            )
        
        
            assert (dialog[-1]['role'] == "user"), \
            f"Last message must be from user, got {dialog[-1]['role']}"
        

            dialog_tokens += self.tokenizer.encode( # 加上对话中的最后一条用户问。
                f"{E_INST} {dialog[-1]['content'].strip()} {E_INST}",
                bos = True,
                eos = True
            )
            
            prompt_tokens.append(dialog_tokens)
        
        
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        ) 
        # generation_tokens.shape = [batch_size, max_gen_len]
        # generation_logprobs.shape = [batch_size, max_gen_len]
        
        if logprobs:
            return [
                {
                    "generation":{
                        "role": "assistant",
                        "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests # shape = (batch_size, )
                )
            ]
            
            
        else:
            return [
                {
                    "generation":{
                        "role": "assistant",
                        "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                    }   
                }
                
                for t, unsafe in zip(generation_tokens, unsafe_requests)
            ]
        
        



































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
    
    
    
    
    
    