# 防御ViT对抗攻击项目

## 项目简介

本项目实现了一个针对Vision Transformer (ViT)的对抗攻击防御系统，通过引入可学习的扰动抑制模块和对抗训练策略，有效提升模型在对抗样本下的鲁棒性，同时保持对干净样本的较高精度。

## 核心技术

### 1. 扰动抑制模块 (Perturbation Suppression Module)
- **原理**: 通过卷积的局部平滑特性过滤高频对抗噪声
- **实现**: 使用轻量级卷积网络，结合残差连接保留原始语义
- **特点**: 可学习的去噪滤波器，自适应抑制微小扰动

### 2. 对抗训练 (Adversarial Training)
- **策略**: 联合训练干净样本和对抗样本
- **损失函数**: `loss = α * loss_clean + (1-α) * loss_adv`
- **优化**: 对抑制模块使用较高学习率，主干网络使用较低学习率

### 3. 白盒PGD攻击
- **方法**: Projected Gradient Descent
- **参数**: ε=8/255, α=2/255, 迭代次数=10
- **用途**: 生成对抗样本进行训练和评估

## 项目结构

```
defence_vit/
├── models.py              # 模型定义（扰动抑制模块、防御ViT）
├── train.py               # 训练脚本（基线模型、防御模型）
├── attack.py              # PGD攻击实现
├── evaluate.py            # 鲁棒性和延迟评估
├── dataset.py             # CIFAR-10数据加载
├── inference.py           # 推理和评估流程
├── visualize.py           # 可视化防御效果
├── requirements.txt       # 依赖包列表
└── readme.md             # 项目说明文档
```

## 环境配置

### 依赖包
```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch 2.10.0
- torchvision 0.25.0
- timm (用于ViT模型)
- matplotlib (可视化)
- captum (可解释性)

### 系统要求
- Python 3.8+
- CUDA 12.x (推荐使用GPU加速)
- 至少8GB内存

## 使用方法

### 1. 训练模型

```bash
python train.py
```

训练过程包括：
- **基线模型训练**: 标准训练，保存为 `base_vit_weights.pth`
- **防御模型训练**: 对抗训练，保存为 `defended_vit_weights.pth`

### 2. 运行推理评估

```bash
python inference.py
```

评估内容：
- 干净样本精度对比
- 对抗样本鲁棒性对比
- 推理延迟分析
- 生成可视化报告

### 3. 查看可视化结果

运行 `inference.py` 后会生成：
- `defense_success_and_tradeoff.png`: 防御成功案例和精度权衡图
- `defense_visualization.png`: 防御效果可视化

## 核心代码说明

### 扰动抑制模块

```python
class PerturbationSuppressionModule(nn.Module):
    def __init__(self):
        super(PerturbationSuppressionModule, self).__init__()
        self.filter = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * 0.5 + self.filter(x) * 0.5
```

### 防御ViT

```python
class DefendedViT(nn.Module):
    def __init__(self, num_classes=10):
        super(DefendedViT, self).__init__()
        self.suppression_module = PerturbationSuppressionModule()
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        purified_x = self.suppression_module(x)
        out = self.vit(purified_x)
        return out
```

## 实验结果

### 性能指标
- **干净样本精度**: 基线模型 vs 防御模型
- **对抗样本精度**: 基线模型 vs 防御模型
- **推理延迟**: 单样本推理时间对比

### 防御效果
- 有效抑制PGD攻击
- 保持较高的干净样本精度
- 可接受的计算开销

## 技术亮点

1. **轻量级设计**: 扰动抑制模块参数量小，计算开销低
2. **端到端训练**: 联合优化去噪模块和分类网络
3. **可解释性**: 可视化展示扰动抑制效果
4. **实用性强**: 可直接集成到现有ViT模型中

