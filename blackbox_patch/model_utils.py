import torch
import os
import config

def load_target_model(device):
    """
    加载 CIFAR-10 预训练的 ResNet20 模型
    """
    print(f"[*] 正在加载模型至缓存目录: {config.MODEL_DIR}")
    # 设置 torch hub 的缓存目录
    torch.hub.set_dir(config.MODEL_DIR)
    
    # 从 chenyaofo 的仓库加载预训练模型
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True)
    
    model = model.to(device)
    model.eval() # 攻击时模型必须处于评估模式，且不更新其参数
    
    # 冻结模型参数
    for param in model.parameters():
        param.requires_grad = False
        
    print("[*] 模型加载完成并已冻结参数。")
    return model