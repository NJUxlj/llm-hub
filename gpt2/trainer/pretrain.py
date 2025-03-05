import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Config, get_scheduler
import wandb
from tqdm.auto import tqdm
from accelerate import Accelerator
from datasets import load_dataset
import sys
sys.path.append("..")
from model.gpt2 import GPT2LMHeadModel
from data.dataset import PretrainingDataset
from utils.training_utils import set_seed, save_checkpoint
from utils.logging_utils import get_logger

logger = get_logger(__name__)

class GPT2Trainer:
    def __init__(self, config):
        """
        初始化GPT2预训练器
        Args:
            config: 训练配置对象，包含所有必要的超参数
        """
        self.config = config
        self.accelerator = Accelerator()
        set_seed(config.seed)
        
        # 初始化模型
        model_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_positions=config.max_position_embeddings,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            activation_function=config.activation_function,
            resid_pdrop=config.resid_pdrop,
            embd_pdrop=config.embd_pdrop,   
            attn_pdrop=config.attn_pdrop,
            layer_norm_epsilon=config.layer_norm_epsilon
        )
        self.model = GPT2LMHeadModel(model_config)
        
        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_epsilon,
            weight_decay=config.weight_decay
        )
        
        # 初始化数据集
        train_dataset = PretrainingDataset(
            dataset_path=config.dataset_path,
            tokenizer_path=config.tokenizer_path,
            max_length=config.max_length
        )
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        
        # 计算总训练步数
        num_update_steps_per_epoch = len(self.train_dataloader)
        self.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        
        # 学习率调度器
        self.lr_scheduler = get_scheduler(
            name=config.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=self.max_train_steps
        )
        
        # 使用accelerator准备所有组件
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
            )
        
        # 初始化wandb
        if self.accelerator.is_main_process:
            wandb.init(
                project=config.wandb_project,
                name=config.run_name,
                config=vars(config)
            )

    def train(self):
        """执行预训练过程"""
        logger.info("Starting pretraining...")
        progress_bar = tqdm(
            range(self.max_train_steps),
            disable=not self.accelerator.is_local_main_process
        )
        completed_steps = 0
        
        for epoch in range(self.config.num_train_epochs):
            self.model.train()
            total_loss = 0
            
            for step, batch in enumerate(self.train_dataloader):
                # 前向传播
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                loss = outputs.loss
                total_loss += loss.detach().float()
                
                # 反向传播
                self.accelerator.backward(loss)
                if self.config.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                
                progress_bar.update(1)
                completed_steps += 1
                
                # 记录日志
                if self.accelerator.is_main_process and step % self.config.logging_steps == 0:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": self.lr_scheduler.get_last_lr()[0],
                        "epoch": epoch,
                        "step": completed_steps,
                    })
                
                # 保存检查点
                if (step + 1) % self.config.save_steps == 0:
                    self.save_model(epoch, step)
                
                if completed_steps >= self.max_train_steps:
                    break
            
            avg_loss = total_loss / len(self.train_dataloader)
            logger.info(f"Epoch {epoch}: Average loss = {avg_loss}")
            
            # 每个epoch结束后保存模型
            self.save_model(epoch, "epoch_end")
    
    def save_model(self, epoch, step):
        """
        保存模型检查点
        Args:
            epoch: 当前训练轮数
            step: 当前步数或标识符
        """
        if self.accelerator.is_main_process:
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            save_path = os.path.join(
                self.config.output_dir,
                f"checkpoint-epoch{epoch}-step{step}"
            )
            save_checkpoint(
                model=unwrapped_model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                epoch=epoch,
                step=step,
                loss=None,
                path=save_path,
                config=self.config
            )
            logger.info(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    from config.model_config import PretrainingConfig
    
    # 加载配置
    config = PretrainingConfig()
    
    # 初始化训练器并开始训练
    trainer = GPT2Trainer(config)
    trainer.train()

