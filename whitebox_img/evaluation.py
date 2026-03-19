import torch
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim

def normalize(img_tensor):
    """对图像进行 CIFAR-10 标准化，对齐模型训练时的输入分布"""
    device = img_tensor.device
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
    return (img_tensor - mean) / std

def evaluate_attack(model, clean_images, adv_images, labels, lpips_loss_fn, device):
    """计算 ASR, LPIPS, SSIM，并返回预测结果用于可视化。"""
    model.eval()
    
    norm_clean = normalize(clean_images)
    norm_adv = normalize(adv_images)

    # 1. 计算攻击成功率 (ASR) 和获取预测标签
    with torch.no_grad():
        clean_outputs = model(norm_clean)
        clean_preds = clean_outputs.argmax(dim=1)
        
        adv_outputs = model(norm_adv)
        adv_preds = adv_outputs.argmax(dim=1)
        
        correct_mask = (clean_preds == labels)
        total_correct_clean = correct_mask.sum().item()
        
        clean_acc = total_correct_clean / labels.size(0)
        
        if total_correct_clean == 0:
            asr = 0.0
        else:
            successful_attacks = (adv_preds[correct_mask] != labels[correct_mask]).sum().item()
            asr = successful_attacks / total_correct_clean

    # 2. 计算 LPIPS
    clean_images_lpips = clean_images * 2.0 - 1.0
    adv_images_lpips = adv_images * 2.0 - 1.0
    
    with torch.no_grad():
        lpips_dist = lpips_loss_fn(clean_images_lpips, adv_images_lpips).mean().item()

    # 3. 计算 SSIM
    clean_np = clean_images.cpu().numpy().transpose(0, 2, 3, 1)
    adv_np = adv_images.cpu().numpy().transpose(0, 2, 3, 1)
    
    ssim_val = 0.0
    for i in range(clean_np.shape[0]):
        val = ssim(clean_np[i], adv_np[i], data_range=1.0, channel_axis=-1)
        ssim_val += val
    ssim_val /= clean_np.shape[0]

    # === 新增：返回 clean_preds 和 adv_preds 用于可视化 ===
    return asr, lpips_dist, ssim_val, clean_acc, clean_preds, adv_preds