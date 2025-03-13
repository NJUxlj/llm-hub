import os  
from transformers import (
    AutoTokenizer, 
    TrainingArguments,
    TrainerCallback, 
    TrainerState, 
    TrainerControl  
)
from datasets import load_dataset  
from trl import SFTTrainer  

# 假设 transformers 库中 GitHub models 目录下已经有 modeling_grok1.py，  
# 并且其中定义了 Grok1ForCausalLM 类  
from ..model.modeling_grok1 import Grok1ModelForCausalLM  
from ..configs.config import MODEL_PATH, DEEPSEEK_CONFIG_PATH


from ..evaluation.evaluate import Evaluator



# ==========================  
# 自定义回调，将评估集成到trainer中  
# ==========================  
class QAEvaluationCallback(TrainerCallback):  
    def __init__(self, eval_dataset, tokenizer, model, evaluator):  
        self.eval_dataset = eval_dataset  
        self.tokenizer = tokenizer  
        self.model = model  
        self.evaluator = evaluator  

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):  
        predictions = []  
        references = []  
        # 遍历验证集，对每个样本进行生成  
        for example in self.eval_dataset:  
            inputs = self.tokenizer(example["text"], return_tensors="pt", truncation=True, max_length=512)  
            input_ids = inputs.input_ids.to(self.model.device)  
            outputs = self.model.generate(input_ids, max_length=128)  
            predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
            if "Answer:" in predicted_text:  
                answer = predicted_text.split("Answer:")[-1].strip()  
            else:  
                answer = predicted_text.strip()  
            predictions.append(answer)  
            references.append(example["ground_truth"])  
        # 调用Evaluator计算指标  
        metrics = self.evaluator.evaluate(predictions, references)  
        print("Integrated Evaluation Metrics:", metrics)  
        # 可将指标写入到 state 或 logs 中  
        state.log_history.append(metrics)  
        return control  




tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)  
model = Grok1ModelForCausalLM.from_pretrained(MODEL_PATH)  

# 加载一个真实存在且可用的 Hugging Face QA 数据集  
dataset = load_dataset("microsoft/wiki_qa", split="train") 

train_dataset = dataset.select(range(0,500)) 
validation_dataset = dataset.select(range(500,1000))

# 对原始数据中的问答对进行格式转换，生成训练文本。注意特殊标记用以区分文本起始和结束  
def format_qa(example):  
    # 注意检查 dataset 中的字段是否为 'question' 和 'answer'  
    return {  
        "text": f"<|startoftext|>Question: {example['question']}\nAnswer: {example['answer']}<|endoftext|>" , 
        "ground_truth": example["answer"]  
    }  

formatted_train_dataset = train_dataset.map(format_qa)  
formatted_validation_dataset = validation_dataset.map(format_qa)

# DeepSpeed 配置文件路径（请确保该文件存在，可以参考下面给出的示例配置）  
deepspeed_config = DEEPSEEK_CONFIG_PATH 

# 配置训练参数：  
# • deepspeed: 指定 DeepSpeed 配置文件路径  
# • report_to: 添加 Swanlab 上报（Swanlab 会读取 TrainingArguments 里的 log 数据）  
training_args = TrainingArguments(  
    output_dir="./output",  
    per_device_train_batch_size=4,  
    gradient_accumulation_steps=2,  
    learning_rate=2e-5,  
    fp16=True,  
    max_steps=1000,  
    logging_dir="./logs",  
    deepspeed=deepspeed_config,  
    report_to=["swanlab"],  # 确保 Swanlab 相关上报集成已经配置（比如正确的环境变量或 SDK 初始化）  
    evaluation_strategy="steps",  # 设置评估策略，根据实际需求调整评估频率  
    eval_steps=200,  # 每200步进行一次评估  
)  


# 如果需要使用语言模型数据打包，可传入 data_collator（此处构造简单示例，实际使用中可调整）  
from transformers import DataCollatorForLanguageModeling  
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)  

# 初始化 SFTTrainer，使用 packing 功能进行长序列打包  
trainer = SFTTrainer(  
    model=model,  
    tokenizer=tokenizer,  
    args=training_args,  
    train_dataset=formatted_train_dataset,  
    dataset_text_field="text",  
    max_seq_length=512,  
    packing=True,
    data_collator=data_collator
)  



# 初始化Evaluator对象  
evaluator = Evaluator()  
# 增加自定义评估回调，将生成的评价指标直接集成到训练过程中  
trainer.add_callback(QAEvaluationCallback(validation_dataset, tokenizer, model, evaluator))  



# 开始微调  
trainer.train()  