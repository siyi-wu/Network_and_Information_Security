import torch
import torchvision
import torchvision.transforms as transforms
import os

torch.hub.set_dir('./torch')

def get_dataloader(batch_size=32, num_samples=200):
    """加载 CIFAR-10 测试集。"""
    transform = transforms.Compose([
        transforms.ToTensor(), # 保持在 [0, 1] 范围内，便于 LPIPS 和攻击
    ])
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    
    # 取子集加速评估
    subset_indices = list(range(num_samples))
    test_subset = torch.utils.data.Subset(testset, subset_indices)
    
    testloader = torch.utils.data.DataLoader(test_subset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    return testloader

def get_pretrained_model(device):
    """获取在 CIFAR-10 上预训练的 ResNet20 模型。"""
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    model = model.to(device)
    model.eval()
    return model