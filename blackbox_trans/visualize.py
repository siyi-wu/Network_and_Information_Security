import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim_metric

# CIFAR-10 的类别名称
CIFAR_CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img, title=None):
    """反归一化并显示图像"""
    img = img.cpu().clone().detach()
    img[0] = img[0] * 0.2023 + 0.4914
    img[1] = img[1] * 0.1994 + 0.4822
    img[2] = img[2] * 0.2010 + 0.4465
    npimg = img.numpy()
    npimg = np.clip(npimg, 0, 1)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title is not None:
        plt.title(title, fontsize=10)

def calculate_ssim(clean_imgs, adv_imgs):
    """计算批量图像的平均结构相似性 (SSIM)"""
    clean = clean_imgs.cpu().clone().detach().numpy()
    adv = adv_imgs.cpu().clone().detach().numpy()

    # 反归一化到 0-1 范围，匹配视觉感知
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 3, 1, 1)
    std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 3, 1, 1)
    clean = np.clip(clean * std + mean, 0, 1)
    adv = np.clip(adv * std + mean, 0, 1)

    # 转换为 (H, W, C) 格式以适配 skimage
    clean = np.transpose(clean, (0, 2, 3, 1))
    adv = np.transpose(adv, (0, 2, 3, 1))

    total_ssim = 0
    for i in range(len(clean)):
        # win_size=7 适合 CIFAR 的 32x32 小图
        val = ssim_metric(clean[i], adv[i], data_range=1.0, channel_axis=-1, win_size=7)
        total_ssim += val
        
    return total_ssim / len(clean)

def plot_tradeoff_curve(ssim_values, asr_values, eps_values):
    """绘制 ASR vs SSIM 权衡曲线"""
    plt.figure(figsize=(8, 6))
    # 绘制折线图
    plt.plot(ssim_values, asr_values, marker='o', linestyle='-', color='crimson', linewidth=2)
    
    # 为每个点添加 eps 标签
    for i, eps in enumerate(eps_values):
        plt.annotate(f'eps={eps:.3f}', 
                     (ssim_values[i], asr_values[i]), 
                     textcoords="offset points", 
                     xytext=(0, 10), 
                     ha='center',
                     fontsize=9)
        
    plt.title('ASR vs. Perceptual Similarity (SSIM) Trade-off', fontsize=14)
    plt.xlabel('Structural Similarity Index (SSIM) -> Higher is more similar', fontsize=12)
    plt.ylabel('Attack Success Rate (ASR) -> Higher is more successful', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("tradeoff_curve.png", dpi=150)
    print(">>> 权衡曲线已成功保存为: tradeoff_curve.png")

def visualize_attack(clean_images, adv_images, clean_preds, adv_preds, true_labels, num_images=5):
    """可视化攻击效果并保存图片"""
    plt.figure(figsize=(15, 8))
    # 确保不超过批次大小
    num_images = min(num_images, clean_images.size(0))
    
    for i in range(num_images):
        # 1. 原始干净图像
        plt.subplot(3, num_images, i + 1)
        true_class = CIFAR_CLASSES[true_labels[i]]
        clean_class = CIFAR_CLASSES[clean_preds[i]]
        imshow(clean_images[i], title=f"Clean\nTrue: {true_class}\nPred: {clean_class}")
        plt.axis('off')

        # 2. 对抗扰动 (噪声)
        noise = adv_images[i] - clean_images[i]
        noise_norm = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(np.transpose(noise_norm.cpu().detach().numpy(), (1, 2, 0)))
        plt.title("Perturbation")
        plt.axis('off')

        # 3. 对抗样本
        plt.subplot(3, num_images, i + 1 + 2 * num_images)
        adv_class = CIFAR_CLASSES[adv_preds[i]]
        imshow(adv_images[i], title=f"Adversarial\nPred: {adv_class}")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig("attack_visualization.png", dpi=150)
    print(">>> 可视化结果已成功保存为: attack_visualization.png")