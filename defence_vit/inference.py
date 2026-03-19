# inference.py
import torch
import os
from dataset import get_dataloaders
from models import get_base_vit, DefendedViT
from evaluate import evaluate_robustness_and_latency
from attack import pgd_attack
from visualize import visualize_defense_and_tradeoff

def run_inference(base_weights="base_vit_weights.pth", def_weights="defended_vit_weights.pth"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- 开始推理评估，使用设备: {device} ---")

    _, test_loader = get_dataloaders(batch_size=32)

    # 1. 初始化并加载经过标准训练的基线模型
    print("\n[加载基线模型...]")
    base_model = get_base_vit(num_classes=10).to(device)
    if os.path.exists(base_weights):
        base_model.load_state_dict(torch.load(base_weights, map_location=device))
        print(f"-> 成功加载基线模型权重: {base_weights}")
    else:
        print(f"-> 警告：未找到 {base_weights}，将使用随机初始化权重。强烈建议先运行 train.py！")

    # 2. 初始化防御模型并加载对抗训练好的权重
    print("\n[加载防御模型权重...]")
    defended_model = DefendedViT(num_classes=10).to(device)
    if os.path.exists(def_weights):
        defended_model.load_state_dict(torch.load(def_weights, map_location=device))
        print(f"-> 成功加载自定义对抗训练权重: {def_weights}")
    else:
        print(f"-> 警告：未找到 {def_weights}，将使用随机初始化权重。强烈建议先运行 train.py！")

    # 3. 运行评估逻辑
    # 注意：为了演示速度，evaluate.py 中默认 test_batches=4。若要完整测试请在下方传入 test_batches=None
    print("\n[开始评估基线 ViT]")
    base_clean_acc, base_adv_acc, base_latency = evaluate_robustness_and_latency(
        base_model, test_loader, device, is_defended=False, test_batches=10
    )
    
    print("\n[开始评估插入抑制模块的 ViT]")
    def_clean_acc, def_adv_acc, def_latency = evaluate_robustness_and_latency(
        defended_model, test_loader, device, is_defended=True, test_batches=10
    )

    # 4. 打印最终量化评估报告
    print("\n" + "="*50)
    print(" 实验二：量化评估报告汇总 ")
    print("="*50)
    print(f"【精度指标】")
    print(f"  - 干净样本精度: 基线 {base_clean_acc:.2f}% | 防御后 {def_clean_acc:.2f}%")
    print(f"  - 对抗样本精度: 基线 {base_adv_acc:.2f}% | 防御后 {def_adv_acc:.2f}%")
    
    print(f"\n【防御代价权衡 (Trade-off)】")
    acc_drop = base_clean_acc - def_clean_acc
    rob_gain = def_adv_acc - base_adv_acc
    print(f"  - 干净样本精度下降率: {acc_drop:.2f}%") # 对应要求: 干净样本精度下降率 vs 鲁棒性提升度 
    print(f"  - 鲁棒性提升度: {rob_gain:.2f}%") # 对应要求: 干净样本精度下降率 vs 鲁棒性提升度 
    
    print(f"\n【计算开销对比 (Latency)】")
    latency_increase = def_latency - base_latency
    print(f"  - 单样本推理延迟: 基线 {base_latency:.2f}ms | 防御后 {def_latency:.2f}ms") # 对应要求: 计算开销对比(防御引入的推理延迟) [cite: 21]
    print(f"  - 防御引入额外延迟: {latency_increase:+.2f}ms") # 对应要求: 计算开销对比(防御引入的推理延迟) [cite: 21]
    print("="*50)
    
    print("\n正在测试集中寻找完美的防御成功案例...")
    success_found = False
    
    # 遍历少量测试数据，寻找符合条件的图像
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 对整个 Batch 生成对抗样本
        base_model.eval()
        adv_images = pgd_attack(base_model, images, labels, device=device)
        
        with torch.no_grad():
            defended_model.eval()
            
            # 获取基线模型对干净图片的预测
            outputs_base_clean = base_model(images)
            _, preds_base_clean = outputs_base_clean.max(1)
            
            # 获取基线模型对对抗图片的预测 (用于确认攻击是否成功)
            outputs_base_adv = base_model(adv_images)
            _, preds_base_adv = outputs_base_adv.max(1)
            
            # 获取防御模型对对抗图片的预测 (包含了净化模块)
            purified_images = defended_model.suppression_module(adv_images)
            outputs_def_adv = defended_model.vit(purified_images)
            _, preds_def_adv = outputs_def_adv.max(1)
            
        # 寻找条件：基线原图预测对，基线被攻击错，防御模型抵御成功
        for i in range(labels.size(0)):
            if (preds_base_clean[i] == labels[i] and 
                preds_base_adv[i] != labels[i] and 
                preds_def_adv[i] == labels[i]):
                
                print(f"-> 找到符合条件的案例！真实标签: {labels[i].item()}")
                
                visualize_defense_and_tradeoff(
                    clean_img=images[i:i+1], 
                    adv_img=adv_images[i:i+1], 
                    purified_img=purified_images[i:i+1], 
                    true_label=labels[i].item(),
                    base_pred=preds_base_clean[i].item(), 
                    adv_pred=preds_base_adv[i].item(), 
                    def_pred=preds_def_adv[i].item(),
                    base_clean_acc=base_clean_acc, 
                    base_adv_acc=base_adv_acc, 
                    def_clean_acc=def_clean_acc, 
                    def_adv_acc=def_adv_acc
                )
                success_found = True
                break
                
        if success_found:
            break

    if not success_found:
        print("-> 提示：在这个数据批次中没有找到完美的防御成功案例。您可以尝试重新运行或放宽条件。")

    print("推理流程全部完成。")

if __name__ == "__main__":
    run_inference()