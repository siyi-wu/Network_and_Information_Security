import torch
import torch.optim as optim
import torch.nn as nn
from dataset import get_dataloaders, mixup_data, mixup_criterion
from model import get_resnet_model
from trades import trades_loss
from attack import test_robustness

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = get_dataloaders(batch_size=128)
    model = get_resnet_model().to(device)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()
    
    epochs = 10 # 仅作演示，实际可能需要 50-100 epochs
    beta = 6.0  # TRADES 正则化权重
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 1. 应用 Mixup 
            mixed_inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, alpha=1.0)
            
            # 2. 计算 TRADES 损失 (KL divergence 部分) 
            logits, loss_robust = trades_loss(model, mixed_inputs, targets, optimizer, beta=beta)
            
            # 3. 计算 Natural Loss (包含 Mixup 逻辑)
            loss_natural = mixup_criterion(criterion, logits, targets_a, targets_b, lam)
            
            # 4. 总损失
            loss = loss_natural + beta * loss_robust
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(trainloader):.4f}")
        
        # 每 Epoch 评估一次鲁棒性
        test_robustness(model, testloader, device)

if __name__ == '__main__':
    main()