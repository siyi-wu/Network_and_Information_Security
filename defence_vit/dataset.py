# dataset.py
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    # 基础的数据预处理，适配 ViT
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    
    # 训练集和测试集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader