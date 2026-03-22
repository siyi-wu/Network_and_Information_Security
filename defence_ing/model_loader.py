import torch
import config

def load_resnet20():
    torch.hub.set_dir(config.HUB_DIR)
    # 加载预训练模型
    model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=True)
    model = model.to(config.DEVICE)
    model.eval()
    return model