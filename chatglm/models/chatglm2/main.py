from transformers import AutoTokenizer, AutoModel, AutoConfig
from modeling_chatglm2 import ChatGLMForConditionalGeneration
from configuration_chatglm import ChatGLMConfig
from config import MODEL_PATH, MODEL_CONFIG_PATH

model_config = ChatGLMConfig.from_json_file(MODEL_CONFIG_PATH)




# if any(k in MODEL_PATH for k in ("gpt", "opt", "bloom")):
#     padding_side = "left"
# else:
#     padding_side = "right"

# print("padding_side = ", padding_side)


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, config = model_config, use_fast=False,trust_remote_code=True, revision="v1.0")

tokenizer.padding_side = "left"

model = ChatGLMForConditionalGeneration.from_pretrained(MODEL_PATH, config = model_config ,trust_remote_code=True, revision="v1.0",device='cuda').eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)

response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
print(response)



