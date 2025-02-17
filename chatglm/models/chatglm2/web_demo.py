from transformers import AutoModel, AutoTokenizer

from modeling_chatglm2 import ChatGLMForConditionalGeneration

import gradio as gr
import mdtex2html
from utils import load_model_on_gpus