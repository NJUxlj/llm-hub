# llm-hub
- it's unfinished
- 实现了多个主流开源模型的 pre-train + finetune 的流程 (仅使用2000个Question-Answering样本)



## How many models we have
- baichuan
- chatglm
- chatglm2
- chatglm3
- chatglm4
- dbrx
- gemma
- grok1
- llama2
- mixtral
- moss
- qwen
- qwen2


## Important Notes
- `ChatGLM2` only supports transformers version `== 4.41.2`, please downgrade using `pip install transformers==4.41.2`


## Updates
1. `ChatGLM2` is already finished with comprehensive explaination and comment added to the files.
2. now we are working on Qwen and Llama.


## Future Works
- if we have time, a tutorial of Beam Search can be updated. 



## Environment Config
- AutoDL Cloud Platform
  
![env](image/env.png)

- then, make sure to pre-download the model weight (e.g. ChatGLM2-6B on the huggingface) to the local storage (e.g., `/root/autodl-tmp/models/chatglm2`).


## Run
```
cd chatglm/models/chatglm2

python main.py

```



## comparsion between different open-source models
- please check `model_compare.xml`