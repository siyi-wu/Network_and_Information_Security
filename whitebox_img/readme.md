# 白盒图像对抗攻击研究

本项目实现了基于 PGD（Projected Gradient Descent）的白盒图像对抗攻击，并在 CIFAR-10 数据集上使用预训练的 ResNet20 模型进行攻击效果评估。项目通过多维度指标（ASR、LPIPS、SSIM）分析攻击强度与感知质量之间的权衡关系。

## 项目结构

```
whitebox_img/
├── main.py              # 主程序入口
├── dataset_prep.py      # 数据集加载和模型准备
├── attack_gen.py        # PGD 攻击生成
├── evaluation.py        # 攻击效果评估
├── visualization.py     # 结果可视化
├── requirements.txt     # 项目依赖
└── results/             # 实验结果输出目录
    ├── tradeoff_curves.png          # 权衡曲线图
    └── adv_examples_eps_*.png      # 不同 eps 下的对抗样本可视化
```

## 功能模块

### 1. 数据集准备 (dataset_prep.py)
- 加载 CIFAR-10 测试集
- 获取预训练的 ResNet20 模型（来自 `chenyaofo/pytorch-cifar-models`）

### 2. 攻击生成 (attack_gen.py)
- 使用 `torchattacks` 库实现 PGD 攻击
- 支持 L_inf 范数约束
- 可配置攻击强度（eps）、步长（alpha）和迭代步数（steps）

### 3. 效果评估 (evaluation.py)
- **ASR (Attack Success Rate)**: 攻击成功率
- **LPIPS (Learned Perceptual Image Patch Similarity)**: 感知距离（越小越相似）
- **SSIM (Structural Similarity Index)**: 结构相似性（越大越相似）
- **Clean Accuracy**: 干净样本的基线准确率

### 4. 可视化 (visualization.py)
- 绘制 ASR vs LPIPS 和 ASR vs SSIM 的权衡曲线
- 展示原图、对抗样本和放大的扰动噪声
- 显示真实标签和预测结果

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- PyTorch 2.10.0
- torchattacks 3.5.1
- lpips 0.1.4
- scikit-image 0.25.2
- matplotlib 3.10.8

## 使用方法

直接运行主程序：

```bash
python main.py
```

程序将自动：
1. 加载 CIFAR-10 测试集（200 个样本）
2. 加载预训练的 ResNet20 模型
3. 对不同 eps 值（1/255 到 24/255）进行 PGD 攻击
4. 评估攻击效果并计算指标
5. 生成可视化图表并保存到 `results/` 目录

## 实验参数

- **数据集**: CIFAR-10 测试集（200 个样本）
- **模型**: ResNet20（预训练）
- **攻击算法**: PGD（L_inf 范数）
- **攻击强度 (eps)**: 1/255, 2/255, 4/255, 6/255, 8/255, 12/255, 16/255, 20/255, 24/255
- **步长 (alpha)**: eps / 4
- **迭代步数 (steps)**: 10

## 输出结果

运行完成后，`results/` 目录将包含：

1. **tradeoff_curves.png**: 展示攻击成功率与感知质量之间的权衡关系
2. **adv_examples_eps_*.png**: 不同 eps 值下的对抗样本可视化对比图

## 实验结果解读

- **ASR**: 攻击成功率越高，说明攻击越有效
- **LPIPS**: 值越低，说明对抗样本与原图在感知上越相似
- **SSIM**: 值越高，说明对抗样本与原图在结构上越相似
- **权衡关系**: 通常 ASR 越高，LPIPS 越高、SSIM 越低，即攻击强度越大，扰动越明显

## 技术细节

- 图像保持在 [0, 1] 范围内，便于 LPIPS 计算和攻击
- 模型输入使用 CIFAR-10 标准化参数（mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]）
- LPIPS 使用 AlexNet 作为特征提取器
- 扰动可视化采用归一化放大显示
