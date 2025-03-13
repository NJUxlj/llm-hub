import re  
import string  
from collections import Counter  
import numpy as np  


class Evaluator:  
    def __init__(self):  
        self.results = {}  

    def normalize_answer(self, s):  
        """归一化答案：小写、去除标点符号、删除不必要的词和多余的空格"""  
        def remove_articles(text):  
            return re.sub(r"\b(a|an|the)\b", " ", text)  
        def white_space_fix(text):  
            return " ".join(text.split())  
        def remove_punc(text):  
            exclude = set(string.punctuation)  
            return "".join(ch for ch in text if ch not in exclude)  
        def lower(text):  
            return text.lower()  
        return white_space_fix(remove_articles(remove_punc(lower(s))))  

    def compute_exact(self, prediction, ground_truth):  
        return int(self.normalize_answer(prediction) == self.normalize_answer(ground_truth))  

    def compute_f1(self, prediction, ground_truth):  
        pred_tokens = self.normalize_answer(prediction).split()  
        gt_tokens = self.normalize_answer(ground_truth).split()  
        common = Counter(pred_tokens) & Counter(gt_tokens)   #  两个字典的keys产生了交集
        num_same = sum(common.values())  # TP
        if len(pred_tokens) == 0 or len(gt_tokens) == 0:  
            # 如果两个输入都为空，则视为完全匹配  
            return int(pred_tokens == gt_tokens)  
        if num_same == 0:  
            return 0  
        precision = num_same / len(pred_tokens)  
        recall = num_same / len(gt_tokens)  
        return (2 * precision * recall) / (precision + recall)  

    def evaluate(self, predictions, references):  
        """  
        predictions: list of model predictions (字符串)  
        references: list of对应的真实答案 (字符串)  
        返回dict，包含 exact_match 与 f1 分数（百分制）。  
        """  
        total = len(predictions)  
        exacts = []  
        f1s = []  
        for pred, ref in zip(predictions, references):  
            exacts.append(self.compute_exact(pred, ref))  
            f1s.append(self.compute_f1(pred, ref))  
        em_score = np.mean(exacts) * 100  
        f1_score = np.mean(f1s) * 100  
        self.results = {"exact_match": em_score, "f1": f1_score}  
        return self.results  