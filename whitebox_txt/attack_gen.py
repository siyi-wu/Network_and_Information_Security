import torch
import torch.nn.functional as F

def get_word_saliency(model, tokenizer, text, label, device):
    """
    计算输入文本中每个 Token 的梯度显著性 (Saliency)。
    梯度越大的 Token，对模型输出的影响越大，是我们优先替换的目标。
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # 1. 获取 Embedding 层的权重
    embeddings = model.get_input_embeddings()(input_ids)
    embeddings.retain_grad() # 核心：保留对连续 Embedding 的梯度

    # 2. 前向传播 (手动传入 embeddings 而不是 input_ids)
    outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
    loss = F.cross_entropy(outputs.logits, torch.tensor([label]).to(device))

    # 3. 反向传播求梯度
    model.zero_grad()
    loss.backward()

    # 4. 计算每个 Token 梯度的 L2 范数作为显著性分数
    word_grads = embeddings.grad[0] # [seq_len, hidden_size]
    saliency_scores = torch.norm(word_grads, dim=-1) # [seq_len]
    
    return input_ids[0], saliency_scores

def gradient_based_word_swap(model, tokenizer, text, label, device, unk_token="[UNK]", max_swaps=2):
    """
    执行基于梯度的词替换攻击。
    这里使用一种简化的扰动方式：将梯度显著性最高的 N 个词替换为 [UNK] 或 Mask，
    模拟对抗语境下的信息缺失或字符扰乱。
    """
    input_ids, saliency_scores = get_word_saliency(model, tokenizer, text, label, device)
    
    # 忽略特殊 Token ([CLS], [SEP], [PAD])
    special_tokens_mask = [
        t in tokenizer.all_special_ids for t in input_ids.tolist()
    ]
    for i, is_special in enumerate(special_tokens_mask):
        if is_special:
            saliency_scores[i] = -1.0 # 设置为负数，使其不被选中

    # 找出显著性最高的 Top-K 个 Token 的索引
    _, top_indices = torch.topk(saliency_scores, k=min(max_swaps, len(saliency_scores)))
    
    # 替换这些 Token
    adv_input_ids = input_ids.clone()
    unk_id = tokenizer.convert_tokens_to_ids(unk_token)
    
    for idx in top_indices:
        if saliency_scores[idx] > 0: # 确保不是特殊 Token
            adv_input_ids[idx] = unk_id
            
    # 解码回文本
    adv_text = tokenizer.decode(adv_input_ids, skip_special_tokens=True)
    return adv_text