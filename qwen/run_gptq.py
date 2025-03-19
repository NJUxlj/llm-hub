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
        
    )
    
    
    tokenizer = AutoTokenizer.from_pretrained()
    
    tokenizer.pad_token_id = tokenizer.eod_id
    
    
    data = preprocess()
    
    
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_name_or_path, 
        quantize_config, 
        device_map="auto", 
        trust_remote_code=True
    )
    
    
    
    logging.basicConfig(
        
    )
    
    
    
    model.quantize()
    
    
    
    model.save_quantized()
    
    
    tokenizer.save_quantized()