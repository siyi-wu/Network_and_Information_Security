# 黑盒对抗补丁攻击 (Blackbox Patch Attack)

基于 CIFAR-10 数据集的黑盒对抗补丁攻击实现，通过优化对抗补丁将图像分类到目标类别，并分析攻击成功率与感知相似度之间的权衡关系。

## 项目概述

本项目实现了一种黑盒对抗补丁攻击方法，针对 CIFAR-10 数据集上的预训练 ResNet20 模型。通过训练不同尺寸的对抗补丁，将图像分类到指定的目标类别，并评估攻击成功率（ASR）和感知相似度（SSIM）。

## 主要功能

- **对抗补丁训练**：使用 Adam 优化器训练对抗补丁，使其能够欺骗目标模型
- **多尺寸补丁评估**：支持 2x2 到 10x10 多种补丁尺寸，分析不同尺寸的攻击效果
- **攻击成功率评估**：计算补丁在测试集上的攻击成功率（ASR）
- **感知相似度计算**：使用 SSIM（结构相似性指数）评估补丁的隐蔽性
- **权衡曲线可视化**：绘制 ASR vs SSIM 权衡曲线，展示攻击效果与隐蔽性的平衡

## 项目结构

```
blackbox_patch/
├── main.py              # 主程序入口
├── config.py            # 配置文件
├── attack_utils.py      # 补丁攻击器类
├── model_utils.py       # 模型加载工具
├── data_utils.py        # 数据加载工具
├── visualize.py         # 可视化工具
├── requirements.txt     # 依赖包列表
├── outputs/             # 输出目录
│   ├── trained_patch_10x10.png      # 训练好的补丁
│   ├── comparison_*.png              # 原图与被攻击图对比
│   └── tradeoff_curve.png           # ASR vs SSIM 权衡曲线
├── torch/               # PyTorch Hub 缓存目录
└── data/                # CIFAR-10 数据集目录
```

## 依赖环境

- Python 3.x
- PyTorch 2.10.0
- TorchVision 0.25.0
- TorchMetrics
- Matplotlib
- Pillow
- 其他依赖见 `requirements.txt`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

在 `config.py` 中可以修改以下参数：

- `PATCH_SIZES`: 补丁尺寸列表，默认为 [2, 4, 6, 8, 10]
- `TARGET_CLASS`: 目标类别，默认为 0（'plane'）
- `BATCH_SIZE`: 批次大小，默认为 128
- `EPOCHS`: 训练轮数，默认为 5
- `LEARNING_RATE`: 学习率，默认为 0.05

CIFAR-10 类别：
```python
CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
```

## 使用方法

### 运行主程序

```bash
python main.py
```

程序将执行以下步骤：

1. 加载预训练的 ResNet20 模型
2. 准备 CIFAR-10 数据集
3. 遍历不同补丁尺寸进行训练和评估
4. 保存训练好的补丁和对比图
5. 绘制 ASR vs SSIM 权衡曲线

### 输出说明

运行完成后，`outputs/` 目录将包含：

- `trained_patch_10x10.png`: 最大尺寸（10x10）的训练补丁
- `comparison_*.png`: 原图与被攻击图的对比示例
- `tradeoff_curve.png`: 展示不同补丁尺寸下 ASR 与 SSIM 的权衡关系

## 核心算法

### 补丁训练

1. 随机初始化补丁张量（范围 [0, 1]）
2. 将补丁随机应用到训练集图像上
3. 计算模型输出与目标类别的交叉熵损失
4. 使用 Adam 优化器更新补丁参数
5. 将补丁值裁剪到 [0, 1] 范围内

### 评估指标

- **攻击成功率 (ASR)**: 成功将图像分类到目标类别的百分比
- **结构相似性 (SSIM)**: 补丁图像与原图的感知相似度，值越高表示补丁越隐蔽

### 权衡分析

通过绘制 ASR vs SSIM 曲线，可以观察到：
- 补丁尺寸越大，攻击成功率越高
- 补丁尺寸越大，感知相似度越低（越容易被发现）
- 需要在攻击效果和隐蔽性之间找到平衡点

## 技术细节

- **目标模型**: CIFAR-10 预训练 ResNet20（来自 chenyaofo/pytorch-cifar-models）
- **优化器**: Adam
- **损失函数**: 交叉熵损失
- **补丁应用**: 随机位置粘贴到图像上
- **数据范围**: [0, 1]（未标准化，便于可视化）

## 注意事项

- 首次运行会自动下载 CIFAR-10 数据集和预训练模型
- 模型参数被冻结，只优化补丁参数
- 评估时模型处于评估模式
- 建议使用 GPU 加速训练
