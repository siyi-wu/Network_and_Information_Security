# 黑盒迁移攻击 (Blackbox Transfer Attack)

基于替代模型的黑盒对抗攻击实现，结合迁移攻击和查询优化策略，在 CIFAR-10 数据集上评估攻击效果。

## 项目概述

本项目实现了一个完整的黑盒对抗攻击框架，主要包含以下核心功能：

- **替代模型训练**：使用目标模型的伪标签训练替代模型
- **迁移攻击**：通过替代模型生成对抗样本并迁移到目标模型
- **查询优化攻击**：在迁移样本基础上进行查询效率优化
- **可视化评估**：生成 ASR vs SSIM 权衡曲线和攻击效果可视化

## 文件结构

```
blackbox_trans/
├── main.py                 # 主程序入口
├── attacks.py              # 攻击方法实现（PGD、查询优化）
├── models.py               # 模型定义（目标模型、替代模型）
├── dataset.py              # CIFAR-10 数据集加载
├── visualize.py            # 可视化和评估工具
├── requirements.txt        # 依赖项列表
├── attack_visualization.png # 攻击效果可视化结果
├── tradeoff_curve.png      # ASR vs SSIM 权衡曲线
└── readme.md               # 项目说明文档
```

## 环境要求

- Python 3.x
- PyTorch 2.10.0
- CUDA 12.x（可选，用于 GPU 加速）

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：
- torch, torchvision, torchaudio
- matplotlib
- scikit-image
- numpy
- torchattacks

## 使用方法

运行主程序：

```bash
python main.py
```

程序将自动执行以下步骤：

1. 加载 CIFAR-10 数据集
2. 加载预训练的 ResNet20 作为黑盒目标模型
3. 创建 ResNet18 作为替代模型
4. 使用伪标签训练替代模型（10 轮）
5. 评估干净样本的准确率
6. 生成 PGD 迁移攻击样本
7. 执行查询优化攻击
8. 生成 ASR vs SSIM 权衡曲线
9. 可视化攻击效果

## 核心模块说明

### attacks.py

- **pgd_attack**: 增强版 PGD 白盒攻击，支持自定义迭代次数
- **query_efficient_attack**: 查询效率优化的黑盒攻击，在迁移样本基础上叠加随机噪声

### models.py

- **get_target_model**: 加载预训练的 CIFAR-10 ResNet20 模型
- **get_substitute_model**: 创建适配 CIFAR-10 的 ResNet18 替代模型

### dataset.py

- **get_cifar10_dataloaders**: 加载 CIFAR-10 训练集和测试集

### visualize.py

- **calculate_ssim**: 计算批量图像的平均结构相似性
- **plot_tradeoff_curve**: 绘制攻击成功率与感知相似性的权衡曲线
- **visualize_attack**: 生成干净图像、扰动和对抗样本的对比可视化

## 输出结果

程序运行后会生成以下文件：

- **attack_visualization.png**: 展示 6 个样本的干净图像、扰动和对抗样本对比
- **tradeoff_curve.png**: 展示不同扰动预算下的 ASR vs SSIM 权衡关系

## 攻击流程

1. **替代模型训练阶段**
   - 使用目标模型的伪标签训练替代模型
   - 训练 10 轮，使用 Adam 优化器

2. **迁移攻击阶段**
   - 在替代模型上生成 PGD 对抗样本（20 次迭代）
   - 将对抗样本迁移到目标模型进行攻击

3. **查询优化阶段**
   - 在迁移样本基础上叠加随机噪声
   - 最多 50 次查询，优化攻击效果

4. **评估阶段**
   - 计算不同扰动预算下的攻击成功率（ASR）
   - 计算结构相似性（SSIM）评估视觉质量
   - 生成权衡曲线

## 性能指标

- **ASR (Attack Success Rate)**: 攻击成功率 = 1 - 准确率
- **SSIM (Structural Similarity Index)**: 结构相似性，值越高表示图像越相似
- **查询开销**: 平均额外查询次数

## 注意事项

- 首次运行会自动下载 CIFAR-10 数据集和预训练模型
- 建议使用 GPU 加速训练和攻击过程
- 可根据需要调整攻击参数（eps, alpha, iters 等）

## 技术特点

- 结合迁移攻击和查询优化，提高黑盒攻击效率
- 使用 SSIM 评估对抗样本的视觉质量
- 生成权衡曲线帮助理解攻击强度与视觉质量的关系
- 完整的可视化展示攻击效果
