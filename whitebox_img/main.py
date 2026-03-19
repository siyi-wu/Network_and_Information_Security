import torch
import lpips
from tqdm import tqdm
from dataset_prep import get_dataloader, get_pretrained_model
from attack_gen import get_pgd_attacker, generate_adv_images
from evaluation import evaluate_attack
from visualization import plot_tradeoff_curves, plot_adversarial_examples

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")

    print("[*] 正在加载数据和模型...")
    dataloader = get_dataloader(batch_size=32, num_samples=200) 
    model = get_pretrained_model(device)
    
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)

    eps_values_int = [1, 2, 4, 6, 8, 12, 16, 20, 24]
    eps_list = [e / 255.0 for e in eps_values_int]
    
    asr_results, lpips_results, ssim_results = [], [], []

    for eps, eps_int in zip(eps_list, eps_values_int):
        print(f"\n{'-'*40}")
        print(f"[-] 正在测试攻击强度: eps = {eps_int}/255")
        
        alpha = eps / 4 
        atk = get_pgd_attacker(model, eps=eps, alpha=alpha, steps=10)
        
        total_asr, total_lpips, total_ssim, total_clean_acc = 0.0, 0.0, 0.0, 0.0
        batches = 0
        
        for images, labels in tqdm(dataloader, desc=f"Eval eps={eps_int}"):
            images, labels = images.to(device), labels.to(device)
            
            adv_images = generate_adv_images(atk, images, labels)
            
            # === 修改：接收预测标签 ===
            batch_asr, batch_lpips, batch_ssim, clean_acc, clean_preds, adv_preds = evaluate_attack(
                model, images, adv_images, labels, lpips_loss_fn, device
            )
            
            # === 新增：在每种强度的第一个批次截取5张图片进行直观展示 ===
            if batches == 0:
                plot_adversarial_examples(
                    images, adv_images, labels, clean_preds, adv_preds, 
                    eps_val=eps_int, save_dir='./results', num_samples=5
                )
            
            total_asr += batch_asr
            total_lpips += batch_lpips
            total_ssim += batch_ssim
            total_clean_acc += clean_acc
            batches += 1
            
        avg_asr = total_asr / batches
        avg_lpips = total_lpips / batches
        avg_ssim = total_ssim / batches
        avg_clean_acc = total_clean_acc / batches
        
        asr_results.append(avg_asr)
        lpips_results.append(avg_lpips)
        ssim_results.append(avg_ssim)
        
        print(f"    -> [基线] 干净样本准确率: {avg_clean_acc:.2%}")
        print(f"    -> [攻击] ASR: {avg_asr:.2%}, LPIPS: {avg_lpips:.4f}, SSIM: {avg_ssim:.4f}")

    print(f"\n{'-'*40}")
    print("[*] 正在生成密集的权衡曲线...")
    plot_tradeoff_curves(eps_list, asr_results, lpips_results, ssim_results)
    print("[*] 实验一阶段完成！完美曲线与样本可视化图均已生成。")

if __name__ == "__main__":
    main()