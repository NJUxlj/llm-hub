import logging
import sys
import os
from datetime import datetime
from typing import Optional
import wandb

class Logger:
    """自定义日志记录器"""
    
    def __init__(
        self,
        name: str,
        log_dir: str,
        level: int = logging.INFO,
        use_wandb: bool = False,
        project_name: Optional[str] = None
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 设置日志文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        
        # 文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Weights & Biases配置
        self.use_wandb = use_wandb
        if use_wandb and project_name:
            wandb.init(project=project_name, name=f"run_{timestamp}")
    
    def info(self, message: str):
        """记录INFO级别日志"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录WARNING级别日志"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """记录ERROR级别日志"""
        self.logger.error(message)
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """记录训练指标"""
        # 记录到日志文件
        metrics_str = " - ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"Step {step}: {metrics_str}" if step else metrics_str)
        
        # 记录到W&B
        if self.use_wandb:
            wandb.log(metrics, step=step)
    
    def close(self):
        """关闭日志记录器"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
        if self.use_wandb:
            wandb.finish()