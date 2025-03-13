# Large Language Model Hub (LLM-Hub)
- It's still in progress ...
- 对于chatglm, qwen2, gpt2等主流模型，我们在原有的model architecture 文件的基础上实现了 pre-train + finetune 的流程 (仅使用2000个Question-Answering样本)
- 90% of the model codes are copyed from huggingface transformers library but with some modifications and lots of chinese comment to help you understand the model architecture.



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
- gpt2
- t5
- llava
- ViT


## Project Structure
- 这里我们就拿`qwen2`模型的文件夹结构举例，其余所有模型都非常类似


```Plain Text
qwen2
├── README.md
├── configs
│   ├── config.py
│   ├── ds_config.json
|—— models
|   ├── __init__.py
|   ├── modeling_qwen2.py
|   ├── tokenization_qwen2.py
|   ├── configuration_qwen2.py
|—— finetune
|   ├── __init__.py
|   ├── sft_trainer.py
|—— pretrain
|   ├── __init__.py
|   ├── pretrain.py
|—— evaluation
|   ├── __init__.py
|   ├── evaluate.py
|—— utils
|—— main.py
|—— cli_demo.py
|—— web_demo.py

```



## Running Results
- all the training process snapshot and training results are stored in `运行结果截图` folder.


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