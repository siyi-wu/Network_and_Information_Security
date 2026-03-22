import torch
import config
from model_utils import load_target_model
from data_utils import get_dataloaders
from attack_utils import PatchAttacker
from visualize import save_patch, save_comparison, plot_tradeoff_curve

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 运行设备: {device}")

    model = load_target_model(device)
    train_loader, test_loader = get_dataloaders()

    # 用于记录曲线数据的列表
    all_asrs = []
    all_ssims = []

    # 遍历不同的补丁尺寸
    for size in config.PATCH_SIZES:
        attacker = PatchAttacker(model, device, patch_size=size)
        trained_patch = attacker.train_patch(train_loader)
        
        # 评估并获取 ASR 和 SSIM
        asr, avg_ssim = attacker.evaluate(test_loader)
        
        all_asrs.append(asr)
        all_ssims.append(avg_ssim)
        
        # 只保存最大尺寸补丁的图片做示例
        if size == config.PATCH_SIZES[-1]:
            save_patch(trained_patch.detach(), filename=f"trained_patch_{size}x{size}.png")
            
            # 生成几张对比图
            data_iter = iter(test_loader)
            images, labels = next(data_iter)
            images = images.to(device)
            with torch.no_grad():
                orig_outputs = model(images)
                _, orig_preds = orig_outputs.max(1)
                patched_images = attacker.apply_patch(images)
                patch_outputs = model(patched_images)
                _, patch_preds = patch_outputs.max(1)

            for i in range(3):
                if labels[i] != config.TARGET_CLASS:
                    save_comparison(images[i], patched_images[i], orig_preds[i], patch_preds[i], index=f"{size}x{size}_{i}")

    # 绘制 ASR vs SSIM 权衡曲线 
    plot_tradeoff_curve(config.PATCH_SIZES, all_asrs, all_ssims)

if __name__ == "__main__":
    main()