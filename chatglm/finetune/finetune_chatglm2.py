
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from ..data.data_preprocess import preprocess_dataset
import logging
from typing import Dict, List
import wandb

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def train(
    model_path: str,
    dataset_name: str,
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    max_seq_length: int = 2048,
    logging_steps: int = 10,
    save_steps: int = 100,
    warmup_steps: int = 100,
    fp16: bool = True,
    use_wandb: bool = False
):
    """
    Fine-tune ChatGLM2 model
    """
    # Initialize wandb if enabled
    if use_wandb:
        wandb.init(project="chatglm2-finetune")

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
    train_dataset = preprocess_dataset(
        dataset_name=dataset_name,
        tokenizer_path=model_path,
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
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            pad_to_multiple_of=8 if fp16 else None,
            return_tensors="pt",
            padding=True
        ),
    )

    # Start training
    logger.info("Starting training...")
    trainer.train()

    # Save the final model
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    logger.info(f"Training completed. Model saved to {output_dir}")

if __name__ == "__main__":
    train(
        model_path="THUDM/chatglm2-6b",
        dataset_name="squad",  # Replace with your dataset
        output_dir="./chatglm2-finetuned",
        num_train_epochs=3,
        use_wandb=False  # Set to True to enable wandb logging
    )

