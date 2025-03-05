
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
import wandb
from typing import Optional, Dict, Any
import logging
from pathlib import Path

# 导入自定义模块
from ..model.gpt2 import GPT2LMHeadModel
from ..data.dataset import GPT2Dataset
from ..utils.training_utils import get_linear_schedule_with_warmup
from ..config.model_config import GPT2Config

class GPT2FineTuner(pl.LightningModule):
    '''
    使用 PyTorch Lightning 进行训练流程管理
    支持混合精度训练(FP16)以提高训练效率
    实现梯度裁剪和梯度累积
    支持断点续训
    使用 WandB 进行实验追踪
    实现早停机制
    支持多GPU训练
    
    '''
    def __init__(
        self,
        model_config: GPT2Config,
        learning_rate: float = 2e-5,
        warmup_steps: int = 1000,
        weight_decay: float = 0.01,
        gradient_clip_val: float = 1.0,
        accumulate_grad_batches: int = 1
    ):
        """
        GPT2模型微调器
        
        Args:
            model_config: 模型配置对象
            learning_rate: 学习率
            warmup_steps: 预热步数
            weight_decay: 权重衰减
            gradient_clip_val: 梯度裁剪阈值
            accumulate_grad_batches: 梯度累积步数
        """
        super().__init__()
        self.save_hyperparameters()
        
        # 初始化模型
        self.model = GPT2LMHeadModel(model_config)
        
        # 训练参数
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        模型前向传播
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

    def training_step(self, batch, batch_idx):
        """
        训练步骤
        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        验证步骤
        """
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"]
        )
        
        loss = outputs.loss
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        """
        配置优化器和学习率调度器
        """
        # 创建优化器
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            eps=1e-8
        )
        
        # 创建学习率调度器
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

def train(
    train_dataset: GPT2Dataset,
    val_dataset: GPT2Dataset,
    model_config: GPT2Config,
    output_dir: str,
    batch_size: int = 8,
    max_epochs: int = 10,
    learning_rate: float = 2e-5,
    warmup_steps: int = 1000,
    weight_decay: float = 0.01,
    gradient_clip_val: float = 1.0,
    accumulate_grad_batches: int = 1,
    fp16: bool = True,
    project_name: str = "gpt2-finetuning",
    **kwargs
) -> None:
    """
    GPT2模型微调主函数
    
    Args:
        train_dataset: 训练数据集
        val_dataset: 验证数据集
        model_config: 模型配置
        output_dir: 输出目录
        batch_size: 批次大小
        max_epochs: 最大训练轮数
        learning_rate: 学习率
        warmup_steps: 预热步数
        weight_decay: 权重衰减
        gradient_clip_val: 梯度裁剪阈值
        accumulate_grad_batches: 梯度累积步数
        fp16: 是否使用混合精度训练
        project_name: WandB项目名称
    """
    # 设置输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 初始化数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 初始化模型
    model = GPT2FineTuner(
        model_config=model_config,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches
    )
    
    # 设置回调函数
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="gpt2-{epoch:02d}-{val_loss:.4f}",
            save_top_k=3,
            monitor="val_loss",
            mode="min"
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            mode="min"
        )
    ]
    
    # 初始化WandB logger
    wandb_logger = WandbLogger(
        project=project_name,
        name=f"gpt2-finetuning-{wandb.util.generate_id()}",
        log_model=True
    )
    
    # 初始化训练器
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",  # 自动检测可用设备
        devices="auto",      # 自动检测设备数量
        strategy="ddp" if torch.cuda.device_count() > 1 else None,  # 多GPU时使用DDP
        precision=16 if fp16 else 32,  # 是否使用混合精度训练
        callbacks=callbacks,
        logger=wandb_logger,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        log_every_n_steps=10,
        **kwargs
    )
    
    # 开始训练
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader
    )
    
    # 保存最终模型
    model.model.save_pretrained(output_dir / "final_model")
    
    logging.info(f"Training completed. Final model saved to {output_dir / 'final_model'}")

if __name__ == "__main__":
    # 示例使用
    from ..config.model_config import get_default_config
    from ..data.dataset import load_datasets
    
    # 加载配置
    config = get_default_config()
    
    # 加载数据集
    train_dataset, val_dataset = load_datasets()
    
    # 开始训练
    train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model_config=config,
        output_dir="outputs",
        batch_size=8,
        max_epochs=10,
        learning_rate=2e-5,
        project_name="gpt2-finetuning"
    )

