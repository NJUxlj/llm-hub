
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import logging
from typing import Dict, List
import wandb

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def prepare_pretrain_data(
    dataset_name: str,
    tokenizer,
    max_seq_length: int = 2048,
    cache_dir: str = None
):
    """
    Prepare dataset for pre-training
    """
    # Load dataset
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_special_tokens_mask=True
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names,
        num_proc=4
    )

    return tokenized_dataset

def pretrain(
    model_path: str,
    dataset_name: str,
    output_dir: str,
    num_train_epochs: int = 1,
    per_device_train_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    max_seq_length: int = 2048,
    logging_steps: int = 10,
    save_steps: int = 1000,
    warmup_steps: int = 1000,
    fp16: bool = True,
    use_wandb: bool = False
):
    """
    Pre-train ChatGLM2 model
    """
    if use_wandb:
        wandb.init(project="chatglm2-pretrain")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # Prepare dataset
    train_dataset = prepare_pretrain_data(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length
    )

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        fp16=fp16,
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        prediction_loss_only=True
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        ),
    )

    # Start training
    logger.info("Starting pre-training...")
    trainer.train()

    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Pre-training completed. Model saved to {output_dir}")

if __name__ == "__main__":
    pretrain(
        model_path="THUDM/chatglm2-6b",
        dataset_name="wikitext",  # Replace with your pre-training dataset
        output_dir="./chatglm2-pretrained",
        num_train_epochs=1,
        use_wandb=False
    )

