import torch
from datasets import load_dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Adafactor
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer

# 参数配置
MODEL_NAME = "google/flan-t5-base"
DATASET_NAME = "squad_v2"  # 问答数据集示例
LORA_RANK = 8
MAX_LENGTH = 512
BATCH_SIZE = 8
GRAD_ACCUM_STEPS = 4

# 初始化组件
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# 添加LoRA适配器
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=LORA_RANK,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"]  # T5的注意力模块
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# 数据集预处理
def format_qa(example):
    input_text = f"question: {example['question']} context: {example['context']}"
    target_text = example["answers"]["text"][0]  # 取第一个答案
    return {"input": input_text, "target": target_text}

def tokenize_fn(examples):
    model_inputs = tokenizer(
        examples["input"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    labels = tokenizer(
        examples["target"],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

dataset = load_dataset(DATASET_NAME)
dataset = dataset.map(format_qa).map(tokenize_fn, batched=True)

# 训练参数
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=3e-4,
    num_train_epochs=3,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=50,
    predict_with_generate=True,
    report_to="tensorboard",
    optim="adafactor",
    warmup_ratio=0.1,
    fp16=True
)

# 初始化训练器
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    dataset_text_field="input",
    max_seq_length=MAX_LENGTH,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
)

# 开始训练
trainer.train()