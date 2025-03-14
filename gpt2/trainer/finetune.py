import os  
import torch  
from transformers import (  
    GPT2LMHeadModel,  
    Trainer,  
    TrainingArguments,  
    EarlyStoppingCallback,  
    default_data_collator  
)  
from swanlab import SwanLabCallback  
from typing import Optional, Dict, Any  
import logging  
from pathlib import Path  

# 自定义模块导入  
from ..data.dataset import GPT2Dataset  
from ..config.model_config import GPT2Config  

class CustomTrainer(Trainer):  
    """自定义优化器实现分组权重衰减"""  
    def create_optimizer(self):  
        # 分组参数设置  
        no_decay = ["bias", "LayerNorm.weight"]  
        optimizer_grouped_parameters = [  
            {  
                "params": [p for n, p in self.model.named_parameters()   
                          if not any(nd in n for nd in no_decay)],  
                "weight_decay": self.args.weight_decay,  
            },  
            {  
                "params": [p for n, p in self.model.named_parameters()   
                          if any(nd in n for nd in no_decay)],  
                "weight_decay": 0.0,  
            },  
        ]  
        
        return torch.optim.AdamW(  
            optimizer_grouped_parameters,  
            lr=self.args.learning_rate,  
            eps=self.args.adam_epsilon  
        )  

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
    # 初始化模型  
    model = GPT2LMHeadModel(model_config)  
    
    # 训练参数配置  
    training_args = TrainingArguments(  
        output_dir=output_dir,  
        num_train_epochs=max_epochs,  
        per_device_train_batch_size=batch_size,  
        per_device_eval_batch_size=batch_size,  
        warmup_steps=warmup_steps,  
        weight_decay=weight_decay,  
        gradient_accumulation_steps=accumulate_grad_batches,  
        fp16=fp16,  
        evaluation_strategy="epoch",  
        save_strategy="epoch",  
        logging_strategy="steps",  
        logging_steps=10,  
        load_best_model_at_end=True,  
        metric_for_best_model="eval_loss",  
        greater_is_better=False,  
        report_to="none",  # 禁用默认监控  
        dataloader_num_workers=4,  
        dataloader_pin_memory=True,  
        gradient_clip_val=gradient_clip_val,  
        **kwargs  
    )  
    
    # 初始化SwanLab回调  
    swanlab_callback = SwanLabCallback(  
        project=project_name,  
        experiment_name=f"gpt2-{os.environ.get('SWANLAB_ID')}",  
        config={  
            "learning_rate": learning_rate,  
            "batch_size": batch_size,  
            "model_type": "gpt2"  
        }  
    )  
    
    # 初始化训练器  
    trainer = CustomTrainer(  
        model=model,  
        args=training_args,  
        train_dataset=train_dataset,  
        eval_dataset=val_dataset,  
        data_collator=default_data_collator,  
        callbacks=[  
            swanlab_callback,  
            EarlyStoppingCallback(early_stopping_patience=3)  
        ]  
    )  
    
    # 开始训练  
    trainer.train()  
    
    # 保存最终模型  
    trainer.save_model(Path(output_dir) / "final_model")  
    logging.info(f"训练完成，模型已保存至 {Path(output_dir)/'final_model'}")  

if __name__ == "__main__":  
    # 示例使用  
    from ..config.model_config import get_default_config  
    from ..data.dataset import load_datasets  
    
    # 初始化配置  
    config = get_default_config()  
    train_dataset, val_dataset = load_datasets()  
    
    # 启动训练  
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