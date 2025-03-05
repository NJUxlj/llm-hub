import torch
import logging
from model import TorchModel
from peft_tuner import (
    get_peft_model, 
    LoraConfig, 
    PromptTuningConfig, 
    PrefixTuningConfig, 
    PromptEncoderConfig,
    PeftType,
    TaskType
)
from .evaluate import Evaluator
from .config import Config


logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#大模型微调策略
tuning_tactics = Config["tuning_tactics"]

print("正在使用 %s"%tuning_tactics)



if tuning_tactics == "lora_tuning":
    peft_config = LoraConfig(
        r = 8,
        lora_alpha=32,lora_dropout=0.1,
        target_modules=["query", "key", "value"],
    )
elif tuning_tactics == "prompt_tuning":
    peft_config = PromptTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=10)
elif tuning_tactics == "prefix_tuning":
    peft_config = PrefixTuningConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=10)
elif tuning_tactics == "p_tuning":
    peft_config = PromptEncoderConfig(task_type=TaskType.SEQ_CLS, num_virtual_tokens=10)
    
    
    
    
model = TorchModel


model = get_peft_model(model, peft_config)

state_dict = model.state_dict()   # 一定要先用 peft模型包裹后 再把权重取出， 不然的话， 一会儿本地的lora权重加不进去

#将微调部分权重加载
if tuning_tactics == "lora_tuning":
    loaded_weight = torch.load('output/lora_tuning.pth')
elif tuning_tactics == "p_tuning":
    loaded_weight = torch.load('output/p_tuning.pth')
elif tuning_tactics == "prompt_tuning":
    loaded_weight = torch.load('output/prompt_tuning.pth')
elif tuning_tactics == "prefix_tuning":
    loaded_weight = torch.load('output/prefix_tuning.pth')
    
    
    
print(loaded_weight.keys())
state_dict.update(loaded_weight)


model.load_state_dict(state_dict)

model = model.cuda()
evaluator = Evaluator(Config, model, logger)

evaluator.eval(0)
