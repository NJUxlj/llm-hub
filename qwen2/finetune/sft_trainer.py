import os
import torch
import argparse
import deepspeed
import swanlab
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import get_peft_model, LoraConfig, AdaLoraConfig, prepare_model_for_kbit_training
from datasets import load_dataset
import configs.config as config  # 导入本地配置

# 增强的错误处理装饰器
def error_handler(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            swanlab.log({"error": str(e)})
            raise
    return wrapper

# 参数解析增强
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", default="data/sft.json", type=str)
    parser.add_argument("--sft_type", choices=["full", "lora", "ada-lora"], default="full")
    parser.add_argument("--deepspeed", default="ds_config.json", type=str)
    parser.add_argument("--quant", choices=["4bit", "8bit", "none"], default="none")  # 新增量化选项
    return parser.parse_args()

@error_handler
def main():
    args = parse_args()
    
    # SwanLab监控增强
    swanlab.init(
        experiment_name=f"qwen2.5-{args.sft_type}-finetune",
        config={
            "model": config.MODEL_PATH,
            "sft_type": args.sft_type,
            "quantization": args.quant,
            "deepspeed_config": args.deepspeed,
        }
    )

    # 量化配置
    bnb_config = None
    if args.quant != "none":
        load_in = {"4bit": 4, "8bit": 8}[args.quant]
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(load_in == 4),
            load_in_8bit=(load_in == 8),
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    # 模型加载增强（从本地路径）
    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_PATH,
        revision="main",
        trust_remote_code=True,
        use_fast=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_PATH,
        revision="main",
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True
    )

    # 量化准备
    if args.quant != "none":
        model = prepare_model_for_kbit_training(model)

    # 适配器配置增强
    if args.sft_type != "full":
        peft_config = {
            "lora": LoraConfig(
                r=8,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 扩展目标模块
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
                modules_to_save=["embed_tokens", "lm_head"]  # 保持embedding层可训练
            ),
            "ada-lora": AdaLoraConfig(
                init_r=12,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
                task_type="CAUSAL_LM",
                target_rank=8,
                final_rank=4
            )
        }[args.sft_type]
        
        model = get_peft_model(model, peft_config)
        model.enable_input_require_grads()  # 支持梯度检查点
        model.print_trainable_parameters()

    # 数据处理增强
    def process_func(examples):
        MAX_LENGTH = 2048  # 扩展上下文长度
        template = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n{answer}<|im_end|>"
        )
        
        texts = [
            template.format(question=q, answer=a)
            for q, a in zip(examples["question"], examples["answer"])
        ]
        
        # 更智能的截断处理
        model_inputs = tokenizer(
            texts,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            add_special_tokens=False  # 手动添加EOS
        )
        
        # 添加EOS token并创建labels
        for i in range(len(model_inputs["input_ids"])):
            if model_inputs["input_ids"][i][-1] != tokenizer.eos_token_id:
                model_inputs["input_ids"][i].append(tokenizer.eos_token_id)
                model_inputs["attention_mask"][i].append(1)
                
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        return model_inputs

    # 数据集加载增强
    dataset = load_dataset("json", 
                         data_files=args.dataset_path,
                         split="train",
                         streaming=False).map(
        process_func,
        batched=True,
        batch_size=1000,
        remove_columns=["question", "answer"]
    )

    # 训练参数增强
    training_args = TrainingArguments(
        output_dir=f"./output/{args.sft_type}",
        per_device_train_batch_size=4,  # 调整为更保守的值
        gradient_accumulation_steps=8,
        learning_rate=2e-5,
        num_train_epochs=3,
        fp16=(not args.quant),  # 量化时自动禁用
        bf16=(args.quant and torch.cuda.is_bf16_supported()),
        logging_steps=50,
        report_to="swanlab",
        deepspeed=args.deepspeed,
        optim="adamw_torch",
        max_grad_norm=0.3,  # 添加梯度裁剪
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="no",
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=(args.sft_type == "full")  # 全量微调时启用
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,  # 对齐显存优化
            return_tensors="pt"
        ),
    )

    # 训练后保存逻辑
    trainer.train()
    
    # 模型保存增强
    save_path = f"output/{args.sft_type}_final"
    if args.sft_type == "full":
        model.save_pretrained(save_path, safe_serialization=True)
    else:
        model.save_pretrained(save_path, safe_serialization=True)
    
    swanlab.log({"save_path": save_path})

if __name__ == "__main__":
    main()