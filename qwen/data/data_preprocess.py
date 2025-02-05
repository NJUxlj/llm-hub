import json
from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer

def preprocess_function(examples, tokenizer: PreTrainedTokenizer, max_length=512):
    # 构造模型输入文本
    input_texts = []
    for conv in examples["conversations"]:
        text = tokenizer.apply_chat_template(conv, tokenize=False)
        input_texts.append(text)
    
    # Tokenization
    tokenized = tokenizer(
        input_texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    
    # 对于因果语言模型，labels应与input_ids相同
    tokenized["labels"] = tokenized["input_ids"].clone()
    
    return tokenized

def process_data(input_file, output_file, tokenizer):
    # 加载原始数据（假设为JSON格式）
    with open(input_file) as f:
        raw_data = json.load(f)
    
    # 转换为Dataset对象
    dataset = Dataset.from_dict({"conversations": raw_data})
    
    # 应用预处理
    processed_dataset = dataset.map(
        preprocess_function,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=["conversations"]
    )
    
    # 保存处理后的数据
    processed_dataset.save_to_disk(output_file)

if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen-7B",
        trust_remote_code=True,
        pad_token="<|endoftext|>"
    )
    
    process_data(
        input_file="raw_data.json",
        output_file="processed_data",
        tokenizer=tokenizer
    )