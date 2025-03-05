# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import torch.nn as nn
import logging
from .config import Config
from .model import TorchModel, choose_optimizer
from .evaluate import Evaluator
from .loader import load_data

from peft import (
    get_peft_model, 
    LoraConfig,
    PromptTuningConfig, 
    PrefixTuningConfig, 
    PromptEncoderConfig,
    TaskType
)

logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
模型训练主程序
"""


seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False






def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)
    #加载模型
    model = TorchModel
    
    
    tuning_tactics = config['tuning_tactics']
    
    if tuning_tactics == "lora_tuning":
        peft_config = LoraConfig(
            r = 8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"]
        )
    elif tuning_tactics == "p_tuning":
        peft_config = PromptEncoderConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=10)
    elif tuning_tactics == "prompt_tuning":
        peft_config = PromptTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=10)
    elif tuning_tactics == "prefix_tuning":
        peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=10)

    model = get_peft_model(model, peft_config)
    
    
    # 防止lora在训练时， classifier被冻结
    if tuning_tactics == "lora_tuning":
        for param in model.get_submodule("model").get_submodule("classifier").parameters():
            param.requires_grad = True
            
            
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    
    optimizer = choose_optimizer(config, model)
    evaluator = Evaluator(config, model, logger)
    
    
    for epoch in range(config['epoch']):
        epoch +=1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        
        for index, batch in enumerate(train_data):
            if cuda_flag:
                batch = [d.cuda() for d in batch]
            
            optimizer.zero_grad()
            input_ids, labels = batch
            output = model.forward(input_ids)[0] # pooled logits shape = (B, num_labels)
            
            loss = nn.CrossEntropyLoss()(output, labels.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
            
            if index % (len(train_data)//2) ==0:
                logger.info("batch loss %f" % loss)
        
        logger.info("epoch average loss %f" % (np.mean(train_loss)))
        acc = evaluator.eval(epoch)
        
    model_path = os.path.join(config["model_path"], "%s.pth" % tuning_tactics)
    save_tunable_parameters(model, model_path)

    return acc
        
            
            
    
    
            


def save_tunable_parameters(model:nn.Module, path):
    saved_params = {
        k:v.to("cpu")
        for k,v in model.named_parameters()
    }
    
    torch.save(saved_params, path)





if __name__ == "__main__":
    main(Config)