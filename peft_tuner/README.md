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





---



方法 3：将相对导入改为绝对导入
如果你希望直接运行 main.py，可以将相对导入改为绝对导入。

在 main.py 中，将：

python
from .config import Config  
改为：

python
from peft.config import Config  
然后直接运行：

bash
python main.py  
注意：

如果你在其他目录运行脚本（例如 llm-hub），需要确保 PYTHONPATH 包含 llm-hub。
方法 4：调整 sys.path（不推荐）
如果你不想修改导入方式，也不想切换目录，可以在 main.py 中手动调整 sys.path。

在 main.py 的顶部添加以下代码：

python
import sys  
import os  

# 将 llm-hub 添加到 sys.path  
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  
然后将相对导入改为绝对导入：

python
from peft.config import Config  
直接运行：

bash
python main.py  
注意：

这种方法虽然可以解决问题，但修改 sys.path 是一种临时解决方案，不推荐在正式项目中使用。