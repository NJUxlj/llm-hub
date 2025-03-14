import re
import numpy as np
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import bert_score


import torch  
from scipy.special import softmax  
from datasets import load_dataset 


class QAEvaluator:
    def __init__(self, model_name="Qwen/Qwen2-7B"):
        """
        初始化评估器
        Args:
            model_name: 要评估的模型名称/路径
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 初始化评估指标
        self.exact_match = evaluate.load("exact_match")
        self.f1 = evaluate.load("f1")
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.bleu = evaluate.load("bleu")
        
        # 检查BERTScore是否可用
        try:
            from bert_score import BERTScorer
            self.bertscorer = BERTScorer(lang="en")
        except ImportError:
            print("BERTScore not installed, skipping...")
            self.bertscorer = None

    def preprocess_qa(self, example: Dict) -> Dict:
        """
        预处理QA数据格式
        Args:
            example: 包含context, question, answers的字典
        Returns:
            处理后的输入字典
        """
        input_text = f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer:"
        return {
            "input": input_text,
            "references": example['answers']['text']  # 支持多参考答案
        }

    def normalize_text(self, text: str) -> str:
        """
        文本标准化处理：
        1. 转换为小写
        2. 移除标点符号
        3. 移除多余空格
        """
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return ' '.join(text.split())

    def calculate_metrics(self, predictions: List[str], references: List[List[str]]) -> Dict:
        """
        计算所有QA指标
        Args:
            predictions: 模型预测答案列表
            references: 参考答案列表（每个问题可能有多个参考答案）
        Returns:
            包含所有指标的字典
        """
        metrics = {}
        
        # Exact Match (考虑多参考答案)
        em_scores = []
        for pred, refs in zip(predictions, references):
            max_em = max(self.exact_match.compute(predictions=[pred], references=[ref])['exact_match']
                        for ref in refs)
            em_scores.append(max_em)
        metrics['exact_match'] = np.mean(em_scores)
        
        # F1 Score (考虑多参考答案)
        f1_scores = []
        for pred, refs in zip(predictions, references):
            max_f1 = max(self.f1.compute(predictions=[pred], references=[ref], average='macro')['f1']
                        for ref in refs)
            f1_scores.append(max_f1)
        metrics['f1'] = np.mean(f1_scores)
        
        # ROUGE-L
        rouge_scores = []
        for pred, refs in zip(predictions, references):
            max_rouge = max(self.rouge.score(pred, ref)['rougeL'].fmeasure for ref in refs)
            rouge_scores.append(max_rouge)
        metrics['rougeL'] = np.mean(rouge_scores)
        
        # BLEU-4
        bleu_scores = []
        for pred, refs in zip(predictions, references):
            tokenized_pred = self.normalize_text(pred).split()
            tokenized_refs = [self.normalize_text(ref).split() for ref in refs]
            bleu_scores.append(sentence_bleu(tokenized_refs, tokenized_pred))
        metrics['bleu'] = np.mean(bleu_scores)
        
        # BERTScore (如果可用)
        if self.bertscorer:
            P, R, F1 = self.bertscorer.score(predictions, [refs[0] for refs in references])
            metrics['bert_precision'] = P.mean().item()
            metrics['bert_recall'] = R.mean().item()
            metrics['bert_f1'] = F1.mean().item()
            
        return metrics

    def evaluate(self, dataset: List[Dict], batch_size: int = 8) -> Dict:
        """
        执行完整评估流程
        Args:
            dataset: 包含context/question/answers的字典列表
            batch_size: 批处理大小
        Returns:
            包含所有评估指标的字典
        """
        processed_data = [self.preprocess_qa(ex) for ex in dataset]
        
        # 生成预测
        predictions = []
        for i in range(0, len(processed_data), batch_size):
            batch = processed_data[i:i+batch_size]
            inputs = self.tokenizer(
                [ex['input'] for ex in batch],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.model.device)
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=4,
                early_stopping=True
            )
            
            batch_preds = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            predictions.extend([p.split("\n")[-1].strip() for p in batch_preds])
        
        # 收集参考答案
        references = [ex['references'] for ex in processed_data]
        
        return self.calculate_metrics(predictions, references)
    
    
    




class MCQEvaluator:  
    def __init__(self, model_name="Qwen/Qwen2-7B"):  
        """  
        初始化MCQ评估器  
        Args:  
            model_name: 要评估的模型名称/路径  
        """  
        self.model = AutoModelForCausalLM.from_pretrained(  
            model_name,  
            trust_remote_code=True,  
            torch_dtype=torch.bfloat16  
        ).eval()  
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)  
        self.tokenizer.pad_token = self.tokenizer.eos_token  
        
        # 初始化指标存储  
        self.metrics = {  
            "accuracy": 0.0,  
            "confidence": 0.0,  
            "calibration_error": 0.0,  
            "distractors_analysis": {}  
        }  

    def format_mcq_prompt(self, example: Dict) -> Dict:  
        """  
        标准化MCQ输入格式（支持RACE/SciQ/CommonsenseQA）  
        Args:  
            example: 包含context/question/options/answer的字典  
        Returns:  
            格式化后的输入字典  
        """  
        prompt_template = (  
            "Read the following and answer the question:\n"  
            "{context}\n\n"  
            "Question: {question}\n"  
            "Options:\n{options}\n"  
            "Answer:"  
        )  
        
        # 处理不同数据集格式  
        options = "\n".join([f"{chr(65+i)}) {opt}"   
                           for i, opt in enumerate(example['options'])])  
        context = example.get('context', '')  
        
        return {  
            "formatted_input": prompt_template.format(  
                context=context,  
                question=example['question'],  
                options=options  
            ),  
            "correct_answer": example['answer']  # 正确选项索引（0-based）  
        }  

    def get_option_logits(self, formatted_input: str) -> List[float]:  
        """  
        获取每个选项的logit分数（基于首字母匹配）  
        """  
        option_letters = ['A', 'B', 'C', 'D', 'E'][:len(self.current_options)]  
        logits = []  
        
        # 并行计算所有选项  
        inputs = [  
            formatted_input + f" {letter})"  
            for letter in option_letters  
        ]  
        
        # 批量编码  
        encoded = self.tokenizer(  
            inputs,  
            return_tensors="pt",  
            padding=True,  
            truncation=True,  
            max_length=512  
        ).to(self.model.device)  
        
        with torch.no_grad():  
            outputs = self.model(**encoded)  
        
        # 提取每个选项的logits（对应字母后的第一个token）  
        for i, letter in enumerate(option_letters):  
            letter_token_id = self.tokenizer.encode(  
                f" {letter})",   
                add_special_tokens=False  
            )[0]  
            logits.append(outputs.logits[i, -1, letter_token_id].item())  
            
        return logits  

    def calculate_calibration_error(self, confidences: List, corrects: List) -> float:  
        """  
        计算预期校准误差（Expected Calibration Error）  
        """  
        bin_boundaries = np.linspace(0, 1, 11)  
        bin_acc = np.zeros_like(bin_boundaries)  
        bin_conf = np.zeros_like(bin_boundaries)  
        bin_count = np.zeros_like(bin_boundaries)  

        for conf, correct in zip(confidences, corrects):  
            bin_idx = np.digitize(conf, bin_boundaries) - 1  
            bin_acc[bin_idx] += correct  
            bin_conf[bin_idx] += conf  
            bin_count[bin_idx] += 1  

        ece = 0.0  
        for i in range(len(bin_boundaries)):  
            if bin_count[i] > 0:  
                acc = bin_acc[i] / bin_count[i]  
                conf = bin_conf[i] / bin_count[i]  
                ece += (abs(acc - conf) * bin_count[i])  
                
        return ece / sum(bin_count)  

    def evaluate(self, dataset: List[Dict], batch_size: int = 8) -> Dict:  
        """  
        执行完整评估流程  
        Args:  
            dataset: 包含context/question/options/answer的字典列表  
            batch_size: 批处理大小  
        Returns:  
            包含所有评估指标的字典  
        """  
        processed_data = [self.format_mcq_prompt(ex) for ex in dataset]  
        all_confidences = []  
        all_corrects = []  
        
        for item in processed_data:  
            self.current_options = item['options']  # 保存当前选项供后续分析  
            logits = self.get_option_logits(item["formatted_input"])  
            probs = softmax(logits)  
            
            # 记录预测结果  
            predicted = np.argmax(probs)  
            is_correct = (predicted == item["correct_answer"])  
            
            # 更新指标  
            self.metrics["accuracy"] += is_correct  
            all_confidences.append(probs[predicted])  
            all_corrects.append(is_correct)  
            
            # 干扰项分析  
            if not is_correct:  
                distractors = [i for i in range(len(probs))   
                              if i != item["correct_answer"]]  
                for d in distractors:  
                    self.metrics["distractors_analysis"].setdefault(d, 0)  
                    self.metrics["distractors_analysis"][d] += (probs[d] > probs[item["correct_answer"]])  
        
        # 计算最终指标  
        total = len(processed_data)  
        self.metrics["accuracy"] /= total  
        self.metrics["confidence"] = np.mean(all_confidences)  
        self.metrics["calibration_error"] = self.calculate_calibration_error(  
            all_confidences, all_corrects  
        )  
        
        # 干扰项分析标准化  
        for k in self.metrics["distractors_analysis"]:  
            self.metrics["distractors_analysis"][k] /= total  
            
        return self.metrics  






# 示例用法
if __name__ == "__main__":
    # 示例数据集（实际应使用SQuAD等标准数据集）
    sample_data = [
        {
            "context": "Paris is the capital of France.",
            "question": "What is the capital of France?",
            "answers": {"text": ["Paris", "The capital is Paris"]}
        }
    ]
    
    evaluator = QAEvaluator()
    results = evaluator.evaluate(sample_data)
    print("Evaluation Results:", results)
    
    print("===========================================")
    
    # 示例数据（CommonsenseQA格式）  
    sample_data = [{  
        "context": "When making a sandcastle, you need to",  
        "question": "What tool is most useful?",  
        "options": ["shovel", "hammer", "thermometer", "paintbrush", "ruler"],  
        "answer": 0  # 正确选项索引  
    }]  
    
    evaluator = MCQEvaluator()  
    results = evaluator.evaluate(sample_data)  
    print("Evaluation Results:", results) 