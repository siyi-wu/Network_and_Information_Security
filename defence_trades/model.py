import torch
import os

def get_resnet_model(device, model_type='resnet20', pretrained=True):
    hub_dir = './torch'
    if not os.path.exists(hub_dir):
        os.makedirs(hub_dir)
    torch.hub.set_dir(hub_dir)
    
    try:
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", 
                               f"cifar10_{model_type}", 
                               pretrained=pretrained,
                               trust_repo=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", f"cifar10_{model_type}", pretrained=pretrained)

    model = model.to(device)
    model.eval() 
    return model