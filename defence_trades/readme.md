# 深度学习模型鲁棒性防御研究

本项目实现了基于TRADES和Mixup的深度学习模型鲁棒性防御方法，并在CIFAR-10数据集上进行了实验验证。

## 项目概述

本项目旨在研究深度学习模型的对抗鲁棒性，通过结合TRADES（Trade-off between Accuracy and Robustness）和Mixup数据增强技术，提升模型在对抗攻击下的表现，同时评估防御代价和计算开销。

## 核心功能

- **标准模型训练与评估**: 使用预训练的ResNet20模型作为基准
- **鲁棒性增强训练**: 结合Mixup和TRADES进行对抗训练
- **对抗攻击测试**: 使用PGD（Projected Gradient Descent）白盒攻击评估模型鲁棒性
- **性能可视化**: 生成精度权衡图和推理延迟对比图

## 文件结构

```
defence_trades/
├── main.py              # 主程序入口
├── model.py             # 模型加载（ResNet20）
├── dataset.py           # CIFAR-10数据集加载和Mixup数据增强
├── attack.py            # PGD对抗攻击实现
├── trades.py            # TRADES损失函数
├── train.py             # 训练脚本
├── visualize.py         # 可视化工具
├── requirements.txt     # 依赖包列表
└── readme.md            # 项目说明文档
```

## 技术细节

### 1. 模型架构
- 使用ResNet20作为基础模型
- 从PyTorch Hub加载预训练权重

### 2. 防御方法
- **TRADES**: 通过KL散度最小化自然样本和对抗样本的预测分布差异
- **Mixup**: 数据增强技术，通过线性插值生成混合样本

### 3. 对抗攻击
- **PGD攻击**: 参数设置
  - ε = 8/255 (扰动范围)
  - α = 2/255 (步长)
  - steps = 20 (迭代次数)

### 4. 评估指标
- **干净样本准确率**: 模型在未受攻击样本上的表现
- **鲁棒准确率**: 模型在PGD对抗攻击下的表现
- **防御代价**: 干净样本准确率的下降幅度
- **推理延迟**: 单次推理所需时间（毫秒）

## 依赖环境

- Python 3.x
- PyTorch 2.10.0
- torchvision 0.25.0
- matplotlib 3.10.8
- numpy 2.2.6
- 其他依赖见 [requirements.txt](file:///e:/Github_Project/Network_and_Information_Security/defence_trades/requirements.txt)

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 运行主程序

```bash
python main.py
```

程序将执行以下步骤：

1. **阶段1**: 加载预训练的ResNet20模型并评估基准性能
2. **阶段2**: 使用Mixup+TRADES进行鲁棒性微调训练
3. **阶段3**: 评估鲁棒模型性能并生成可视化结果

### 训练参数

- **数据集**: CIFAR-10
- **批大小**: 128
- **训练轮数**: 15 epochs
- **学习率**: 0.001（微调）
- **TRADES参数**:
  - β = 6.0 (鲁棒性权重)
  - ε = 0.031
  - 步长 = 0.003
  - 扰动步数 = 10

## 输出结果

程序运行后会生成以下文件：

- **tradeoff_evaluation.png**: 干净样本精度与鲁棒性提升的权衡对比图
- **latency_comparison.png**: 标准模型与鲁棒模型的推理延迟对比图

## 实验结果示例

```
Standard Model -> Clean ACC: 92.50%, Robust ACC (PGD): 8.20%
Robust Model -> Clean ACC: 89.30%, Robust ACC (PGD): 45.60%

防御代价 (Accuracy Drop): 3.20%
鲁棒性增益 (Robustness Gain): 37.40%

Standard Model Latency: 0.856 ms
Robust Model Latency:   0.862 ms
```

## 核心代码说明

### [main.py](file:///e:/Github_Project/Network_and_Information_Security/defence_trades/main.py)
- `train_standard()`: 标准模型训练
- `train_robust()`: 鲁棒模型训练（Mixup+TRADES）
- `evaluate()`: 模型评估（干净/对抗样本）

### [trades.py](file:///e:/Github_Project/Network_and_Information_Security/defence_trades/trades.py)
- `trades_loss()`: TRADES损失函数实现

### [attack.py](file:///e:/Github_Project/Network_and_Information_Security/defence_trades/attack.py)
- `pgd_attack()`: PGD对抗攻击实现

### [visualize.py](file:///e:/Github_Project/Network_and_Information_Security/defence_trades/visualize.py)
- `plot_tradeoff()`: 绘制精度权衡图
- `measure_inference_latency()`: 测量推理延迟
- `plot_latency()`: 绘制延迟对比图

## 参考文献

- TRADES: Zhang et al. "Theoretically Grounded Trade-off between Robustness and Accuracy" (ICML 2019)
- Mixup: Zhang et al. "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)

## 注意事项

- 首次运行会自动下载CIFAR-10数据集和预训练模型
- 建议使用GPU加速训练和推理
- 训练过程可能需要较长时间，取决于硬件配置

## 许可证

本项目仅用于学术研究和教育目的。
