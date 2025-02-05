
import json
from typing import Dict, List
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset

class ChatGLM2Dataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        tokenizer: AutoTokenizer,
        max_seq_length: int = 2048
    ):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        question = example["question"]
        answer = example["answer"]

        # Format prompt
        prompt = f"问：{question}\n答："
        
        # Tokenize input
        prompt_ids = self.tokenizer.encode(
            prompt,
            max_length=self.max_seq_length,
            truncation=True,
            add_special_tokens=False
        )
        
        answer_ids = self.tokenizer.encode(
            answer,
            max_length=self.max_seq_length,
            truncation=True,
            add_special_tokens=False
        )

        input_ids = prompt_ids + answer_ids + [self.tokenizer.eos_token_id]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Create labels for training (-100 for prompt tokens)
        labels = [-100] * len(prompt_ids) + answer_ids + [self.tokenizer.eos_token_id]

        # Pad sequences
        padding_length = self.max_seq_length - len(input_ids)
        if padding_length > 0:
            input_ids.extend([self.tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            labels.extend([-100] * padding_length)

        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)
        }

def preprocess_dataset(
    dataset_name: str,
    tokenizer_path: str,
    max_seq_length: int = 2048,
    cache_dir: str = None
) -> ChatGLM2Dataset:
    """
    Preprocess the dataset for ChatGLM2 fine-tuning
    """
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset from Hugging Face
    raw_dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    # Convert dataset to list of dictionaries
    train_data = []
    for example in raw_dataset["train"]:
        # Adapt this according to your dataset structure
        train_data.append({
            "question": example["question"],
            "answer": example["answer"]
        })

    # Create ChatGLM2Dataset
    dataset = ChatGLM2Dataset(
        data=train_data,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length
    )

    return dataset

if __name__ == "__main__":
    # Example usage
    dataset = preprocess_dataset(
        dataset_name="squad",  # Replace with your dataset
        tokenizer_path="THUDM/chatglm2-6b",
        max_seq_length=2048
    )
    print(f"Dataset size: {len(dataset)}")

