# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, PreTrainedTokenizer
"""
数据加载
"""



class DataGenerator:
    
    
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label  = {0: '家居', 1: '房产', 2: '股票', 3: '社会', 4: '文化',
                               5: '国际', 6: '教育', 7: '军事', 8: '彩票', 9: '旅游',
                               10: '体育', 11: '科技', 12: '汽车', 13: '健康',
                               14: '娱乐', 15: '财经', 16: '时尚', 17: '游戏'}
        self.label_to_index = {v:k for k,v in self.index_to_label.items()}
        self.config['class_num'] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer: PreTrainedTokenizer = BertTokenizer.from_pretrained(self.config["pretrain_model_path"])
        
        self.vocab = load_vocab(self.config["vocab_path"])
        
        self.config["vocab_size"] = len(self.vocab)
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line.strip())
                tag = line["tag"]
                title = line["title"]
                label = self.label_to_index[tag]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(
                            text = title,
                            max_length=self.config["max_length"], 
                            pad_to_max_length=True
                        )
                else:
                    input_id = self.encode_sentence(title)
                    
                input_id = torch.LongTensor(input_id)
                label =  torch.LongTensor([label])

                self.data.append([input_id, label])
        return 
        
        
    
    
    def encode_sentence(self, text):
        input_id = []
        
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))

        input_id = self.padding(input_id)
        return input_id
    
    
    
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id = input_id + [0] * (self.config["max_length"] - len(input_id))
        return input_id
    
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        return self.data[index]




def load_vocab(vocab_path):
    token_dict = {}
    with open(file=vocab_path, mode="r", encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置
    return token_dict
    
    
    
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path = data_path, config=config)
    
    dl = DataLoader(dg, batch_size = config['batch_size'], shuffle=shuffle)

    return dl
    





if __name__ == "__main__":
    from .config import Config
    dg = DataGenerator(data_path = Config["valid_data_path"], config=Config)
    
    print(dg[1])