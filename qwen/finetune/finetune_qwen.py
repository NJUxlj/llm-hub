import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk

def load_model_and_tokenizer(model_name="Qwen/Qwen-7B"):
    # 加载模型和tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        pad_token="<|endoftext|>"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto"
    )
    
    return model, tokenizer

def main():
    # 参数配置
    model, tokenizer = load_model_and_tokenizer()
    
    # 加载预处理数据
    dataset = load_from_disk("processed_data")
    dataset = dataset.train_test_split(test_size=0.1)
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir="./qwen-finetuned",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_dir="./logs",
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        deepspeed="./configs/deepspeed_config.json",  # 可选
        report_to="tensorboard"
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
    )
    
    # 开始训练
    trainer.train()
    
    # 保存最终模型
    trainer.save_model("./qwen-finetuned/final")

if __name__ == "__main__":
    main()