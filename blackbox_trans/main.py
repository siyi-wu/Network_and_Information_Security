import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_cifar10_dataloaders
from models import get_target_model, get_substitute_model
from attacks import pgd_attack, query_efficient_attack
from visualize import visualize_attack
from visualize import calculate_ssim, plot_tradeoff_curve

def train_substitute_model(target_model, sub_model, trainloader, device, epochs=10):
    """增加训练轮数至 10 轮，让替代模型更好地逼近目标模型"""
    print(f"开始使用伪标签训练替代模型，共 {epochs} 轮...")
    optimizer = optim.Adam(sub_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    sub_model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, _) in enumerate(trainloader):
            inputs = inputs.to(device)
            
            with torch.no_grad():
                target_outputs = target_model(inputs)
                pseudo_labels = target_outputs.argmax(dim=1)
            
            optimizer.zero_grad()
            sub_outputs = sub_model(inputs)
            loss = criterion(sub_outputs, pseudo_labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if i % 200 == 199:
                print(f"[Epoch {epoch + 1:2d}, Batch {i + 1:3d}] Loss: {running_loss / 200:.4f}")
                running_loss = 0.0
    print("替代模型训练完成！")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用的设备: {device}")
    
    trainloader, testloader = get_cifar10_dataloaders(batch_size=64)
    target_model = get_target_model(device)
    sub_model = get_substitute_model(device)
    
    # 1. 训练更强的替代模型
    train_substitute_model(target_model, sub_model, trainloader, device, epochs=10)
    
    # 2. 评估阶段
    sub_model.eval()
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    # 获取目标模型在干净样本上的预测
    with torch.no_grad():
        clean_outputs = target_model(images)
        clean_preds = clean_outputs.argmax(dim=1)
        clean_acc = (clean_preds == labels).sum().item() / labels.size(0)
    print(f"\n[干净样本] 目标模型准确率: {clean_acc * 100:.2f}%")
    
    # 3. 强化迁移攻击阶段
    print("\n生成 PGD 迁移攻击样本 (iters=20)...")
    adv_images_transfer = pgd_attack(sub_model, images, labels, eps=8/255, alpha=2/255, iters=20)
    
    with torch.no_grad():
        transfer_outputs = target_model(adv_images_transfer)
        transfer_preds = transfer_outputs.argmax(dim=1)
        transfer_acc = (transfer_preds == labels).sum().item() / labels.size(0)
    print(f"[迁移攻击] 目标模型准确率大幅降至: {transfer_acc * 100:.2f}%")
    
    # 4. 优化查询攻击阶段
    print("\n执行基于随机搜索的查询优化攻击...")
    final_adv_images, success_mask, query_counts = query_efficient_attack(
        target_model, adv_images_transfer, labels, max_queries=50
    )
    
    with torch.no_grad():
        final_preds = target_model(final_adv_images).argmax(dim=1)
        final_acc = (final_preds == labels).sum().item() / labels.size(0)
    
    avg_queries = query_counts.float().mean().item()
    print(f"[查询优化后] 目标模型最终准确率: {final_acc * 100:.2f}%")
    print(f"[查询开销] 平均额外查询次数: {avg_queries:.2f} 次/样本")

    # 5. 调用可视化模块
    print("\n" + "="*50)
    print("开始进行 ASR vs SSIM 权衡曲线的量化评估...")
    

    
    # 定义一组不同的扰动预算 eps (对应的 255 分制大约为: 2, 4, 8, 12, 16)
    eps_list = [2/255, 4/255, 8/255, 12/255, 16/255]
    asr_results = []
    ssim_results = []
    
    for current_eps in eps_list:
        print(f"\n评估 eps = {current_eps:.3f} ...")
        
        # 1. 使用当前 eps 生成迁移对抗样本 (由于评估较慢，这里仅用迁移攻击演示趋势)
        current_adv_images = pgd_attack(sub_model, images, labels, eps=current_eps, alpha=current_eps/4, iters=20)
        
        # 2. 计算目标模型上的攻击成功率 (ASR)
        # ASR = (原先预测正确且现在预测错误的样本数) / (原先预测正确的样本数)
        # 为了简便，我们直接使用：1 - 当前准确率 
        with torch.no_grad():
            preds = target_model(current_adv_images).argmax(dim=1)
            # 计算准确率
            acc = (preds == labels).sum().item() / labels.size(0)
            # 攻击成功率 ASR = 1 - 准确率
            asr = 1.0 - acc 
        
        # 3. 计算感知相似性 (SSIM)
        current_ssim = calculate_ssim(images, current_adv_images)
        
        asr_results.append(asr)
        ssim_results.append(current_ssim)
        
        print(f" -> SSIM: {current_ssim:.4f} | ASR: {asr * 100:.2f}%")
        
    # 4. 绘制并保存曲线
    plot_tradeoff_curve(ssim_results, asr_results, eps_list)
    print("="*50 + "\n")
    print("\n正在生成可视化图像...")
    visualize_attack(images, final_adv_images, clean_preds, final_preds, labels, num_images=6)

if __name__ == "__main__":
    main()