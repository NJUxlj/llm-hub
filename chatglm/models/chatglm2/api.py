from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModel
from modeling_chatglm2 import ChatGLMConfig, ChatGLMForConditionalGeneration
import uvicorn, json, datetime
import torch

from typing import Dict

DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


from config import MODEL_PATH


'''

这段代码是一个基于Python的异步HTTP服务端点的实现，通常用于处理前端发送的请求，并返回一个基于输入的生成式AI模型的响应。
代码中使用了FastAPI框架，它是一个现代、快速（高性能）的Web框架，适合构建API。
'''


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache() # 清空 CUDA 缓存
            torch.cuda.ipc_collect() # 收集并释放共享内存


app = FastAPI()

# 异步函数：使用 async def 定义，表示该函数支持异步操作（例如处理 I/O 密集型任务），从而提高性能。
# Request：FastAPI 提供的类，用于处理 HTTP 请求，允许我们从请求中提取数据。
@app.post("/")
async def create_item(request: Request)->Dict:
    global model, tokenizer
    
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    
    prompt = json_post_list.get('prompt')
    history = json_post_list.get('history')
    max_length = json_post_list.get('max_length')
    top_p = json_post_list.get('top_p')
    temperature = json_post_list.get('temperature')
    
    response, history = model.chat(tokenizer,
                                   prompt,
                                   history=history,
                                   max_length=max_length if max_length else 2048,
                                   top_p=top_p if top_p else 0.7,
                                   temperature=temperature if temperature else 0.95)
    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")
    answer = {
        "response": response,
        "history": history,
        "status": 200,
        "time": time
    }
    log = "[" + time + "] " + '", prompt:"' + prompt + '", response:"' + repr(response) + '"'
    print(log)
    torch_gc()
    return answer


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = ChatGLMForConditionalGeneration.from_pretrained(MODEL_PATH, trust_remote_code=True).cuda()
    # 多显卡支持，使用下面三行代替上面两行，将num_gpus改为你实际的显卡数量
    # model_path = "THUDM/chatglm2-6b"
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model = load_model_on_gpus(model_path, num_gpus=2)
    model.eval()
    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)