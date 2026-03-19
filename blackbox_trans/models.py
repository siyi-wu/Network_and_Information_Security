import torch
import torch.nn as nn
import torchvision.models as models

def get_target_model(device):
    """获取黑盒目标模型 ResNet20"""
    print("正在加载目标模型 ResNet20...")
    torch.hub.set_dir('./torch')
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    model.to(device)
    model.eval()
    return model

def get_substitute_model(device):
    """使用修改版的 ResNet18 作为替代模型，增强拟合能力和迁移性"""
    # 不使用预训练权重，从头开始用伪标签训练
    model = models.resnet18(weights=None)
    # 适配 CIFAR-10 的 32x32 输入：将一开始的 7x7 卷积改为 3x3，并去掉最大池化
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10) # CIFAR-10 有 10 个类别
    
    model.to(device)
    return model