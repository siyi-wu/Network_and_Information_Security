import matplotlib.pyplot as plt
import os
import numpy as np

# CIFAR-10 标签映射字典
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']

def plot_tradeoff_curves(eps_list, asr_list, lpips_list, ssim_list, save_dir='./results'):
    """绘制 ASR vs LPIPS 和 ASR vs SSIM 的权衡曲线并保存。"""
    os.makedirs(save_dir, exist_ok=True)
    eps_display = [int(e * 255) for e in eps_list]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(lpips_list, asr_list, marker='o', color='red', linestyle='-', linewidth=2)
    for i, txt in enumerate(eps_display):
        plt.annotate(f'eps={txt}', (lpips_list[i], asr_list[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.title('Trade-off: ASR vs LPIPS')
    plt.xlabel('Perceptual Distance (LPIPS - Lower is better)')
    plt.ylabel('Attack Success Rate (ASR)')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(ssim_list, asr_list, marker='s', color='blue', linestyle='-', linewidth=2)
    for i, txt in enumerate(eps_display):
        plt.annotate(f'eps={txt}', (ssim_list[i], asr_list[i]), textcoords="offset points", xytext=(0,10), ha='center')
    plt.title('Trade-off: ASR vs SSIM')
    plt.xlabel('Structural Similarity (SSIM - Higher is better)')
    plt.ylabel('Attack Success Rate (ASR)')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'tradeoff_curves.png')
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_adversarial_examples(clean_images, adv_images, true_labels, clean_preds, adv_preds, eps_val, save_dir='./results', num_samples=5):
    """
    可视化原图、对抗样本和放大的扰动差异。
    """
    os.makedirs(save_dir, exist_ok=True)
    # 确保不超过批次大小
    num_samples = min(num_samples, clean_images.size(0)) 
    
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 9))
    fig.suptitle(f'Adversarial Examples (PGD Attack, $\epsilon$={eps_val}/255)', fontsize=16)
    
    for i in range(num_samples):
        # 1. 显示干净样本
        ax_clean = axes[0, i]
        clean_img = clean_images[i].cpu().numpy().transpose(1, 2, 0)
        # === 新增：强制截断到 [0, 1] 范围，消除 imshow 警告 ===
        clean_img = np.clip(clean_img, 0, 1) 
        ax_clean.imshow(clean_img)
        ax_clean.axis('off')
        
        true_class = CIFAR10_CLASSES[true_labels[i].item()]
        clean_pred_class = CIFAR10_CLASSES[clean_preds[i].item()]
        color = 'green' if true_labels[i] == clean_preds[i] else 'red'
        ax_clean.set_title(f"True: {true_class}\nClean Pred: {clean_pred_class}", color=color)
        
        # 2. 显示对抗样本
        ax_adv = axes[1, i]
        adv_img = adv_images[i].cpu().numpy().transpose(1, 2, 0)
        # === 新增：强制截断到 [0, 1] 范围，消除 imshow 警告 ===
        adv_img = np.clip(adv_img, 0, 1) 
        ax_adv.imshow(adv_img)
        ax_adv.axis('off')
        
        adv_pred_class = CIFAR10_CLASSES[adv_preds[i].item()]
        color_adv = 'green' if true_labels[i] == adv_preds[i] else 'red'
        ax_adv.set_title(f"Adv Pred: {adv_pred_class}", color=color_adv)
        
        # 3. 显示放大的扰动噪声 (这段保持不变，因为我们已经自己做了归一化处理)
        ax_noise = axes[2, i]
        noise = adv_img - clean_img
        noise_disp = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        ax_noise.imshow(noise_disp)
        ax_noise.axis('off')
        ax_noise.set_title("Perturbation (Amplified)")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'adv_examples_eps_{eps_val}.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"    -> [*] 对抗样本可视化已保存至: {save_path}")