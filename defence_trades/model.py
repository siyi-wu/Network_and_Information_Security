import torch
import os

def get_resnet_model(device, model_type='resnet20', pretrained=True):
    # 1. 强制设定存储路径
    hub_dir = './torch'
    if not os.path.exists(hub_dir):
        os.makedirs(hub_dir)
    torch.hub.set_dir(hub_dir)
    
    # 2. 从指定仓库加载 [cite: 7]
    # 如果 32% 持续出现，尝试删除 ./torch 文件夹重新下载
    try:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", 
                               f"cifar10_{model_type}", 
                               pretrained=pretrained,
                               trust_repo=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        # 备选方案：手动指定类别数
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar10_{model_type}", pretrained=pretrained)

    model = model.to(device)
    model.eval() # 评估模式确保 BatchNorm 行为正确
    return model