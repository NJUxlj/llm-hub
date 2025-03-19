# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple command-line interactive chat demo."""

import argparse
import os
import platform
import shutil
from copy import deepcopy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
from transformers.trainer_utils import set_seed

DEFAULT_CKPT_PATH = 'Qwen/Qwen-7B-Chat'



_WELCOME_MSG = '''\
Welcome to use Qwen-Chat model, type text to start chat, type :h to show command help.
(欢迎使用 Qwen-Chat 模型，输入内容即可进行对话，:h 显示命令帮助。)

Note: This demo is governed by the original license of Qwen.
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, including hate speech, violence, pornography, deception, etc.
(注：本演示受Qwen的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)
'''
_HELP_MSG = '''\
Commands:
    :help / :h          Show this help message              显示帮助信息
    :exit / :quit / :q  Exit the demo                       退出Demo
    :clear / :cl        Clear screen                        清屏
    :clear-his / :clh   Clear history                       清除对话历史
    :history / :his     Show history                        显示对话历史
    :seed               Show current random seed            显示当前随机种子
    :seed <N>           Set random seed to <N>              设置随机种子
    :conf               Show current generation config      显示生成配置
    :conf <key>=<value> Change generation config            修改生成配置
    :reset-conf         Reset generation config             重置生成配置
'''




def _load_model_tokenizer(args):
    pass





def _gc():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        
        

def _clear_screen():
    if platform.system() == 'Windows':
        os.system('cls')
    else:
        os.system('clear')
        
        
def _print_history():
    pass



def _get_input():
    pass





def main():
    parser = argparse.ArgumentParser(
        description='QWen-Chat command-line interactive chat demo.')
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("-s", "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")
    args = parser.parse_args()
    
    




if __name__ == "__main__":
    main()