import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from typing import Dict, Optional, Union
import os

class TrainingUtils:
    """训练工具类"""
    
    @staticmethod
    def get_optimizer(
        model: nn.Module,
        learning_rate: float,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8
    ) -> torch.optim.Optimizer:
        """
        创建优化器，使用AdamW并应用权重衰减
        """
        # 将参数分为两组：需要和不需要权重衰减的
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=eps
        )
    
    @staticmethod
    def get_scheduler(
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int
    ) -> LambdaLR:
        """
        创建学习率调度器，使用线性预热和余弦衰减
        """
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / \
                      float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            
        return LambdaLR(optimizer, lr_lambda)
    
    @staticmethod
    def save_checkpoint(
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        epoch: int,
        loss: float,
        save_dir: str,
        name: str
    ):
        """
        保存检查点
        """
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss
        }
        
        path = os.path.join(save_dir, f'{name}_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
    
    @staticmethod
    def load_checkpoint(
        path: str,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[LambdaLR] = None
    ) -> Dict[str, Union[int, float]]:
        """
        加载检查点
        """
        checkpoint = torch.load(path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return {
            'epoch': checkpoint['epoch'],
            'loss': checkpoint['loss']
        }
    
    @staticmethod
    def compute_gradient_norm(model: nn.Module) -> float:
        """
        计算梯度范数
        """
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm