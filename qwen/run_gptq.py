import argparse
import json
from typing import Dict
import logging

import torch
import transformers
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
    system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens
    
    # Apply prompt templates
    
    
    





if __name__ == "__main__":
    parser = argparse.ArgumentParser("Model Quantization using AutoGPTQ")
    parser.add_argument("--model_name_or_path", type=str, help="model path")
    parser.add_argument("--data_path", type=str, help="calibration data path")
    parser.add_argument("--out_path", type=str, help="output path of the quantized model")
    parser.add_argument("--max_len", type=int, default=8192, help="max length of calibration data")
    parser.add_argument("--bits", type=int, default=4, help="the bits of quantized model. 4 indicates int4 models.")
    parser.add_argument("--group-size", type=int, default=128, help="the group size of quantized model")
    args = parser.parse_args()
    
    
    quantize_config = BaseQuantizeConfig(
        bits = args.bits,  # 量化位数（4表示4位整型）
        group_size = args.group_size,
        damp_percent=0.01,  # 阻尼系数（用于平滑Hessian矩阵计算）
        # # 禁用激活降序【descendant activations】排列（加速推理但可能轻微影响精度）
        desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad
        static_groups=False, # 使用动态分组替代静态分组
        sym=True,    # 使用对称量化（量化范围对称于0点）
        true_sequential=True,   # 按实际层顺序执行量化
        model_name_or_path=None,   # 模型路径占位符（继承父类参数）
        model_file_base_name="model" # 量化模型文件基础名称
    )
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    
    tokenizer.pad_token_id = tokenizer.eod_id
    
    
    data = preprocess(json.load(open(args.data_path)), tokenizer, args.max_len)
    
    
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_name_or_path, 
        quantize_config, 
        device_map="auto", 
        trust_remote_code=True
    )
    
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )

    model.quantize(data, cache_examples_on_gpu=False)

    model.save_quantized(args.out_path, use_safetensors = True)
    tokenizer.save_quantized(args.out_path)