# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
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


    def load(self):
        self.data = []
        
        
    
    
    def encode_sentence(self):
        pass
    
    
    
    def padding(self):
        pass
    
    
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, index):
        return self.data[index]




def load_vocab(vocab_path):
    token_dict = {}
    
    
    
    
def load_data(data_path, config, shuffle=True):
    pass



