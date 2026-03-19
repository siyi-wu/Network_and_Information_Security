# models.py
import torch
import torch.nn as nn
import timm

class PerturbationSuppressionModule(nn.Module):
    """
    可学习的扰动抑制模块 (轻量级前端去噪) 
    通过卷积的局部平滑特性过滤高频对抗噪声。
    """
    def __init__(self):
        super(PerturbationSuppressionModule, self).__init__()
        self.filter = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid() # 保证输出依然在图像像素的合理区间 [0, 1]
        )

    def forward(self, x):
        # 融合残差连接：保留原始语义，抑制微小扰动
        return x * 0.5 + self.filter(x) * 0.5

class DefendedViT(nn.Module):
    """
    集成防御模块的 ViT
    """
    def __init__(self, num_classes=10):
        super(DefendedViT, self).__init__()
        self.suppression_module = PerturbationSuppressionModule()
        # 使用轻量级 ViT 以适应计算资源
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        # 1. 净化图像
        purified_x = self.suppression_module(x)
        # 2. 传入原生 ViT
        out = self.vit(purified_x)
        return out

def get_base_vit(num_classes=10):
    """获取未加防御的基线模型对比用"""
    return timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)