import torch
import torchvision
import torchvision.transforms as transforms
import config

def get_dataloaders():
    """
    获取 CIFAR-10 数据集。
    我们使用训练集来优化补丁，使用测试集来评估补丁效果。
    """
    print("[*] 正在准备数据集...")
    # CIFAR-10 只转换为 Tensor，不进行标准化，方便补丁直接在 [0,1] 范围内优化和可视化
    transform = transforms.ToTensor()

    train_dataset = torchvision.datasets.CIFAR10(root=config.DATA_DIR, train=True, 
                                                 download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.BATCH_SIZE, 
                                               shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(root=config.DATA_DIR, train=False, 
                                                download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.BATCH_SIZE, 
                                              shuffle=False, num_workers=2)
    
    return train_loader, test_loader