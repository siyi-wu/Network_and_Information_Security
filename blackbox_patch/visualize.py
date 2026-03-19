import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import config
import os

def save_patch(patch_tensor, filename="trained_patch.png"):
    patch_img = TF.to_pil_image(patch_tensor.cpu())
    path = os.path.join(config.OUTPUT_DIR, filename)
    patch_img.resize((128, 128)).save(path)

def save_comparison(original_img, patched_img, original_pred, patched_pred, index=0):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(TF.to_pil_image(original_img.cpu()))
    axes[0].set_title(f"Original\nPred: {config.CLASSES[original_pred]}")
    axes[0].axis('off')
    axes[1].imshow(TF.to_pil_image(patched_img.cpu()))
    axes[1].set_title(f"Patched\nPred: {config.CLASSES[patched_pred]}")
    axes[1].axis('off')
    path = os.path.join(config.OUTPUT_DIR, f"comparison_{index}.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()

def plot_tradeoff_curve(patch_sizes, asrs, ssims):
    """
    绘制 ASR 与 SSIM 的权衡曲线
    """
    plt.figure(figsize=(8, 6))
    
    # 绘制折线图，X轴为SSIM(感知相似度)，Y轴为ASR(攻击成功率)
    plt.plot(ssims, asrs, marker='o', linestyle='-', color='b', markersize=8)
    
    # 在每个数据点旁边标出对应的补丁尺寸
    for i in range(len(patch_sizes)):
        plt.annotate(f"{patch_sizes[i]}x{patch_sizes[i]}", 
                     (ssims[i], asrs[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center')
        
    plt.title('Trade-off: Attack Success Rate vs. Perceptual Similarity (SSIM)')
    plt.xlabel('Perceptual Similarity (SSIM) -> Higher is more invisible')
    plt.ylabel('Attack Success Rate (ASR) %')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 调整坐标轴方向（通常习惯让SSIM从大到小排列，因为越往右代表越隐蔽）
    plt.gca().invert_xaxis()
    
    path = os.path.join(config.OUTPUT_DIR, "tradeoff_curve.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"\n[*] 权衡曲线已保存至: {path}")