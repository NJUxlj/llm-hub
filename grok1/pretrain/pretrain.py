import os  
from transformers import AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling  
from datasets import load_dataset  
from model.modeling_grok1 import Grok1ModelForCausalLM  


from configs.config import MODEL_PATH, DEEPSEEK_CONFIG_PATH

# =====================================  
# 加载预训练数据集：使用小规模的wikitext数据集  
# =====================================  
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")  
train_dataset = dataset["train"]  


model_path = MODEL_PATH 
tokenizer = AutoTokenizer.from_pretrained(model_path)  
model = Grok1ModelForCausalLM.from_pretrained(model_path)  

# =====================================  
# 数据预处理：Tokenize 和 chunk 处理  
# =====================================  
def tokenize_function(example):  
    return tokenizer(example["text"])  

# 对文本进行tokenize  
tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])  

# 将长文本拼接后切分为max_seq_length的片段  
max_seq_length = 512  
def group_texts(examples):  
    # 将所有tokens拼接起来  
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}  
    total_length = len(concatenated[list(examples.keys())[0]])  
    # 截断到可被max_seq_length整除的长度  
    if total_length >= max_seq_length:  
        total_length = (total_length // max_seq_length) * max_seq_length  
    result = {  
        k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]  
        for k, t in concatenated.items()  
    }  
    return result  

grouped_dataset = tokenized_dataset.map(group_texts, batched=True)  

# 配置DataCollator，关闭MLM，使用自回归预测  
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  

# =====================================  
# 配置训练参数：集成DeepSpeed与Swanlab上报  
# =====================================  
training_args = TrainingArguments(  
    output_dir="./output/pretrain",  
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=2,  
    learning_rate=5e-5,  
    fp16=True,  
    max_steps=1000,  
    logging_dir="./logs/pretrain",  
    deepspeed="ds_config.json",  # 请确保该文件存在且配置正确  
    report_to=["swanlab"],        # 确保Swanlab监控已经配置  
    save_steps=500,  
    evaluation_strategy="no"      # 预训练阶段可以不做评估  
)  

# =====================================  
# 初始化 Trainer 并启动预训练  
# =====================================  
trainer = Trainer(  
    model=model,  
    args=training_args,  
    train_dataset=grouped_dataset,  
    data_collator=data_collator,  
)  

def main():  
    trainer.train()  
    # 保存最终训练模型  
    trainer.save_model("./output/pretrained_grok")  

if __name__ == "__main__":  
    main()  