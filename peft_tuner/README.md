peft子项目，展示了微调的标准流程


## 如何运行
切换到 llm-hub 目录：

bash
cd ..  
使用 -m 参数运行 main.py：

bash
python -m peft_tuner.main 
原因：

使用 -m 参数时，Python 会将 peft 视为 llm-hub 包的一部分，从而正确解析相对导入。`