# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import os
from dataset import get_dataloaders
from models import get_base_vit, DefendedViT
from attack import pgd_attack

def train_base_model(epochs=5, save_path="base_vit_weights.pth"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n--- 1. 开始训练基线模型 (Standard Training) | 设备: {device} ---")

    train_loader, _ = get_dataloaders(batch_size=32)
    model = get_base_vit(num_classes=10).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Base Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} | Clean Acc: {100.*correct/total:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"--- 基线模型训练完成，权重保存至: {save_path} ---")

def train_robust_model(epochs=10, save_path="defended_vit_weights.pth"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n--- 2. 开始对抗训练防御模型 (Joint Adversarial Training) | 设备: {device} ---")

    train_loader, _ = get_dataloaders(batch_size=32)
    model = DefendedViT(num_classes=10).to(device)
    
    # 针对抑制模块赋予较高的学习率，让它快速学习去噪；对主干网络进行微调
    optimizer = optim.AdamW([
        {'params': model.suppression_module.parameters(), 'lr': 1e-3},
        {'params': model.vit.parameters(), 'lr': 1e-5}
    ], weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    # 权衡参数：控制干净样本精度与鲁棒性的比重
    alpha = 0.6  

    for epoch in range(epochs):
        correct_clean = 0
        correct_adv = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # 第一步：动态生成对抗样本 (使用 PGD)
            model.eval() 
            adv_images = pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, iters=5, device=device)
            model.train() 
            
            # 第二步：联合训练 (Joint Training)
            optimizer.zero_grad()
            
            # 分别计算模型在干净图像和对抗图像上的输出
            outputs_clean = model(images)
            outputs_adv = model(adv_images)
            
            # 分别计算损失
            loss_clean = criterion(outputs_clean, labels)
            loss_adv = criterion(outputs_adv, labels)
            
            # 组合损失函数：通过 alpha 调节两者的比例
            loss = alpha * loss_clean + (1 - alpha) * loss_adv
            
            loss.backward()
            optimizer.step()
            
            # 统计并打印训练指标
            total += labels.size(0)
            _, predicted_clean = outputs_clean.max(1)
            _, predicted_adv = outputs_adv.max(1)
            correct_clean += predicted_clean.eq(labels).sum().item()
            correct_adv += predicted_adv.eq(labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f"Defended Epoch [{epoch+1}/{epochs}] Batch [{batch_idx}/{len(train_loader)}] "
                      f"Total Loss: {loss.item():.4f} | "
                      f"Clean Acc: {100.*correct_clean/total:.2f}% | "
                      f"Adv Acc: {100.*correct_adv/total:.2f}%")

    torch.save(model.state_dict(), save_path)
    print(f"--- 防御模型训练完成，权重保存至: {save_path} ---")

if __name__ == "__main__":
    epochs_to_run = 7
    train_base_model(epochs=epochs_to_run)
    train_robust_model(epochs=epochs_to_run)