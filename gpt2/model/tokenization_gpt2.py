# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for OpenAI GPT."""

import json
import os
from functools import lru_cache
from typing import List, Optional, Tuple, Dict, Any, Callable

import regex as re

from transformers.tokenization_utils import AddedToken, PreTrainedTokenizer
from transformers.utils import logging


logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {
    "vocab_file": "vocab.json",
    "merges_file": "merges.txt",
}




'''
功能：@lru_cache() 是 Python 标准库 functools 提供的一个装饰器，用于对函数的返回值进行缓存。
    LRU 的全称是 Least Recently Used，意思是最近最少使用缓存策略。
    这个装饰器会自动缓存函数的返回值，当函数被多次调用时，如果参数相同，就直接返回缓存的结果，而不用重复计算。
    
    为什么用它？
        这里的 bytes_to_unicode() 函数返回的是一个固定的映射表，每次调用的结果是一样的。为了避免重复计算，使用缓存可以节省时间和内存。
'''
@lru_cache()
def bytes_to_unicode()->Dict[int,str]:
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.

    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    
    
    1. 函数返回的是一个映射表，将 UTF-8 字节（bytes）映射到 Unicode 字符。
    2. 这个映射表避免了空白字符（如空格、换行符）和控制字符（如回车符）的干扰，因为这些字符可能会导致分词算法（比如 BPE）出错。
    3. BPE 分词算法通常在 Unicode 字符串上工作，而不是直接在字节上工作。
    4. 如果数据集非常大（比如 100 亿个 token 的数据集），为了避免出现未知字符（UNK），需要一个较大的 Unicode 字符集来覆盖所有可能的字节。
    
    
    ##Function:
    这段代码的核心功能是构建一个从字节（bytes）到 Unicode 字符的映射表。它的作用是为字节和 Unicode 字符之间提供双向映射，特别是为了支持某些特殊编码需求，比如避免控制字符或空白字符的干扰。
    这种映射在自然语言处理（NLP）中的 Byte Pair Encoding (BPE) 或其他分词算法中非常常见，尤其是在处理大规模语料库时。
    """
    
    '''
    ord("!") 到 ord("~")：对应可打印的 ASCII 字符（从 ! 到 ~）。
    ord("¡") 到 ord("¬")：对应扩展的拉丁字符（Latin-1 Supplement）。
    ord("®") 到 ord("ÿ")：对应更多扩展的拉丁字符。
    '''
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8): # 确保所有可能的字节值（0 到 255）都有对应的 Unicode 映射。
        if b not in bs:
            bs.append(b)
            # 同时，cs 中为这个字节值分配一个新的 Unicode 值：
            # 新的 Unicode 值从 256 开始（2**8），然后递增。
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs] 

    return dict(zip(bs, cs))




def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    
    找到一个单词中的所有符号对

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs



class GPT2Tokenizer(PreTrainedTokenizer):
    """
    Construct a GPT-2 tokenizer. Based on byte-level Byte-Pair-Encoding.

    This tokenizer has been trained to treat spaces like parts of the tokens (a bit like sentencepiece) so a word will
    be encoded differently whether it is at the beginning of the sentence (without space) or not:

    ```python
    >>> from transformers import GPT2Tokenizer

    >>> tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    >>> tokenizer("Hello world")["input_ids"]
    [15496, 995]

    >>> tokenizer(" Hello world")["input_ids"]
    [18435, 995]
    ```

    You can get around that behavior by passing `add_prefix_space=True` when instantiating this tokenizer or when you
    call it on some text, but since the model was not pretrained this way, it might yield a decrease in performance.

    <Tip>

    When used with `is_split_into_words=True`, this tokenizer will add a space before each word (even the first one).

    </Tip>

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        merges_file (`str`):
            Path to the merges file.
        errors (`str`, *optional*, defaults to `"replace"`):
            Paradigm to follow when decoding bytes to UTF-8. See
            [bytes.decode](https://docs.python.org/3/library/stdtypes.html#bytes.decode) for more information.
        unk_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The beginning of sequence token.
        eos_token (`str`, *optional*, defaults to `"<|endoftext|>"`):
            The end of sequence token.
        pad_token (`str`, *optional*):
            The token used for padding, for example when batching sequences of different lengths.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (GPT2 tokenizer detect beginning of words by the preceding space).
        add_bos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial beginning of sentence token to the input. This allows to treat the leading
            word just as any other word.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        merges_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        add_bos_token=False,
        **kwargs,
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        self.add_bos_token = add_bos_token

        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        
        super().__init__(
            errors=errors,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,    
            **kwargs,
        )
        
    @property
    def vocab_size(self):
        return len(self.encoder)
    
    
    def get_vocab(self):
        return dict(self.encoder, **self.added_tokens_encoder)
    
    
    
    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)
    
    
    
    
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if self.add_bos_token:
            bos_token_ids = [self.bos_token_id]
        else:
            bos_token_ids = []

        output = bos_token_ids + token_ids_0

        if token_ids_1 is None:
            return output

        return output + bos_token_ids + token_ids_1
    
    
    
    
    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
