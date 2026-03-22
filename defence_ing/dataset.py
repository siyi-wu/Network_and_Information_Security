import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import config
import torch

def get_dataloader():
    # CIFAR-10 标准归一化参数
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root=config.DATA_DIR, train=False, download=True, transform=transform_test)
    
    # 仅取部分数据用于实验，以节省时间
    subset_indices = list(range(500)) 
    testset_subset = torch.utils.data.Subset(testset, subset_indices)
    
    testloader = DataLoader(testset_subset, batch_size=config.BATCH_SIZE, shuffle=False)
    return testloader, testset.classes