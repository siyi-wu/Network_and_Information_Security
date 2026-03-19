import torch
import torch.nn as nn

def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, iters=20):
    """增强版 PGD 白盒攻击：增加了 iters 提升攻击强度"""
    images = images.clone().detach()
    labels = labels.clone().detach()
    original_images = images.clone()
    
    loss_fn = nn.CrossEntropyLoss()
    
    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        adv_images = images + alpha * images.grad.sign()
        
        eta = torch.clamp(adv_images - original_images, min=-eps, max=eps)
        # 针对 normalize 后的图像空间做粗略的裁剪
        images = torch.clamp(original_images + eta, min=-2.5, max=2.5).detach() 
        
    return images.detach()

def query_efficient_attack(target_model, initial_adv_images, true_labels, max_queries=50):
    """
    改进的查询优化：
    在迁移样本的基础上，叠加随机噪声并查询。如果噪声成功改变了预测，则保留该噪声。
    """
    optimized_images = initial_adv_images.clone()
    batch_size = optimized_images.size(0)
    
    success_mask = torch.zeros(batch_size, dtype=torch.bool).to(initial_adv_images.device)
    query_counts = torch.zeros(batch_size).to(initial_adv_images.device)
    
    # 首先检查初始的迁移样本是否已经成功
    with torch.no_grad():
        initial_preds = target_model(optimized_images).argmax(dim=1)
        success_mask = (initial_preds != true_labels)
    
    for q in range(max_queries):
        if success_mask.all():
            break
            
        # 生成候选扰动 (只对尚未成功的样本生效)
        noise = torch.randn_like(optimized_images) * 0.05 
        candidate_images = torch.clamp(optimized_images + noise, min=-2.5, max=2.5)
        
        with torch.no_grad():
            # 模拟黑盒查询
            cand_outputs = target_model(candidate_images)
            cand_preds = cand_outputs.argmax(dim=1)
            
            # 判断候选图像是否攻击成功
            cand_success = (cand_preds != true_labels)
            
            # 找到那些之前没成功，但这次候选图像成功了的样本
            newly_succeeded = cand_success & (~success_mask)
            
            # 仅更新这些成功找到更优边界的样本
            optimized_images[newly_succeeded] = candidate_images[newly_succeeded]
            
            # 更新查询次数和成功掩码
            query_counts[~success_mask] += 1
            success_mask = success_mask | cand_success
            
    return optimized_images, success_mask, query_counts