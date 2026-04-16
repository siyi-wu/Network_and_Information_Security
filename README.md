# 网络与信息安全实验：对抗样本攻击与防御

## 项目简介

本项目实现了 AI 模型的对抗攻击与防御技术，涵盖图像分类（ResNet/ViT）和自然语言处理（BERT）模型的安全攻防测试与量化评估。

## 核心成果

**攻击方面**：
- PGD 白盒攻击在微小扰动（ε=4/255）下达到 **91.88%** 攻击成功率
- 6×6对抗补丁在物理模拟中实现 **75.98%** 攻击成功率，SSIM 达 0.9114

**防御方面**：
- ViT+PSM 模块在仅增加 31.6% 推理延迟下，鲁棒性提升 **22.50%**
- TRADES 对抗训练有效平衡准确率与鲁棒性

## 项目结构

```
.
├── whitebox_img/          # 图像白盒攻击（PGD+ 语义约束）
├── whitebox_txt/          # 文本白盒攻击（梯度扰动）
├── blackbox_trans/        # 黑盒迁移攻击（替代模型）
├── blackbox_patch/        # 黑盒对抗补丁攻击
├── defence_trades/        # TRADES 对抗训练防御
├── defence_vit/           # ViT 扰动抑制模块防御
└── defence_ing/           # 特征归因异常检测防御
```

## 快速开始

### 环境要求
- Python 3.8+
- PyTorch 1.9+
- CUDA 11.8（推荐）

### 安装依赖
```bash
# 进入任意子目录
cd whitebox_img
pip install -r requirements.txt
```

### 运行示例
```bash
# 图像白盒攻击
cd whitebox_img
python main.py

# 黑盒迁移攻击
cd blackbox_trans
python main.py

# TRADES 防御训练
cd defence_trades
python main.py
```

## 实验数据

| 模块 | 数据集 | 模型 | 评估指标 |
|------|--------|------|----------|
| whitebox_img | CIFAR-10 | ResNet20 | ASR/LPIPS/SSIM |
| whitebox_txt | Rotten Tomatoes | BERT | ASR/词修改率 |
| blackbox_trans | CIFAR-10 | ResNet18 | ASR/SSIM |
| blackbox_patch | CIFAR-10 | ResNet20 | ASR/SSIM |
| defence_trades | CIFAR-10 | ResNet20 | 干净准确率/鲁棒准确率 |
| defence_vit | CIFAR-10 | ViT-Tiny | 防御成功率/延迟 |
| defence_ing | CIFAR-10 | ResNet20 | 检测率/误报率 |

## 主要依赖

- **深度学习框架**: PyTorch 2.10.0
- **攻击库**: torchattacks 3.5.1
- **数据集**: CIFAR-10, Rotten Tomatoes
- **可视化工具**: Matplotlib, Seaborn

## 实验环境

- **云平台**: AutoDL 云服务器
- **GPU**: vGPU 32GB
- **CPU**: 10 vCPU Intel Xeon Gold 6459C
- **内存**: 80GB

## 详细文档

各子模块的详细说明请参考对应目录下的 `readme.md` 文件。

## 许可证

本项目仅用于学术研究和教育目的。
