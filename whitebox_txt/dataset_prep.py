import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset

def load_model_and_tokenizer(model_name="textattack/bert-base-uncased-rotten-tomatoes", device="cuda"):
    """
    加载一个在文本分类任务上微调过的 BERT 模型和对应的 Tokenizer。
    """
    print(f"[*] 正在加载模型和 Tokenizer: {model_name}")
    
    # 设置模型保存的同级目录
    model_dir = "./torch"
    os.makedirs(model_dir, exist_ok=True)
    
    # 添加 cache_dir 参数
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_dir)
    # 输出 attention_hidden_states 方便提取 embedding 梯度
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=model_dir)
    
    model.to(device)
    model.eval()
    return model, tokenizer

def load_test_data(num_samples=100):
    """
    加载测试数据集。使用 Rotten Tomatoes 电影评论数据集。
    """
    print(f"[*] 正在加载数据集 (取 {num_samples} 条测试)...")
    
    # 设置数据集保存的同级目录
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 添加 cache_dir 参数
    dataset = load_dataset("rotten_tomatoes", split="test", cache_dir=data_dir)
    
    # 转换为 list 方便处理
    texts = dataset["text"][:num_samples]
    labels = dataset["label"][:num_samples]
    return texts, labels