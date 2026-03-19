import torch
import re

def predict_text(model, tokenizer, text, device):
    """对单条文本进行预测"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        pred_label = outputs.logits.argmax(dim=-1).item()
    return pred_label

def calculate_word_change_rate(original_text, adv_text):
    """计算词汇级别的修改率，作为感知相似性的替代指标"""
    orig_words = set(original_text.lower().split())
    adv_words = set(adv_text.lower().split())
    
    if len(orig_words) == 0:
        return 0.0
        
    changed_words = len(orig_words.symmetric_difference(adv_words))
    rate = changed_words / (len(orig_words) + 1e-5)
    return min(rate, 1.0)

def highlight_adversarial_text(original_text, adv_text, unk_token="[UNK]"):
    """
    在终端中高亮显示对抗文本中的修改部分。
    使用 ANSI 红色转义码突出显示被替换的 token。
    """
    RED = '\033[91m'
    RESET = '\033[0m'
    
    # 将 [UNK] 替换为带有红色高亮的形式
    highlighted_adv = adv_text.replace(unk_token, f"{RED}{unk_token}{RESET}")
    return highlighted_adv