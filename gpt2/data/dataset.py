
import torch
from torch.utils.data import Dataset
from typing import Dict, List
import numpy as np

class GPT2Dataset(Dataset):
    """GPT-2数据集类"""
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 1024,
        stride: int = 512
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # 对文本进行编码
        self.examples = []
        for text in texts:
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                return_overflowing_tokens=True,
                stride=stride,
                return_tensors="pt"
            )
            
            for i in range(len(tokenized["input_ids"])):
                self.examples.append({
                    "input_ids": tokenized["input_ids"][i],
                    "attention_mask": tokenized["attention_mask"][i]
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.examples[i]["input_ids"],
            "attention_mask": self.examples[i]["attention_mask"],
            "labels": self.examples[i]["input_ids"].clone()
        }

