# 白盒文本对抗攻击 (White-box Text Adversarial Attacks)

基于梯度显著性的文本对抗攻击实现，使用BERT模型在Rotten Tomatoes数据集上进行攻击实验。

## 项目概述

本项目实现了一种白盒文本对抗攻击方法，通过计算输入文本中每个token的梯度显著性，识别对模型预测影响最大的词汇，并将其替换为`[UNK]`标记来生成对抗样本。项目包含完整的攻击流程、评估指标和可视化工具。

## 核心功能

- **梯度显著性计算**：通过反向传播计算每个token对模型输出的影响程度
- **基于梯度的词替换攻击**：将梯度显著性最高的词汇替换为`[UNK]`标记
- **多维度评估**：攻击成功率(ASR)、词汇修改率等指标
- **可视化分析**：生成权衡曲线、HTML报告等多种可视化结果

## 文件结构

```
whitebox_txt/
├── main.py                 # 主程序入口
├── attack_gen.py           # 攻击生成模块
├── dataset_prep.py         # 数据集和模型加载
├── evaluation.py           # 评估指标计算
├── visualization.py        # 可视化工具
├── requirements.txt        # 依赖包列表
├── readme.md              # 项目说明文档
└── results/               # 实验结果输出目录
    ├── nlp_tradeoff_curves.png
    ├── perceptual_tradeoff_WordChangeRate_as_Perceptual.png
    └── adversarial_examples_report.html
```

## 环境要求

### 硬件要求
- GPU (推荐，支持CUDA)
- 至少8GB内存

### 软件依赖
- Python 3.8+
- PyTorch 2.10.0+
- Transformers 5.3.0+
- 其他依赖见 [requirements.txt](requirements.txt)

## 安装步骤

1. 克隆项目仓库
```bash
git clone <repository_url>
cd whitebox_txt
```

2. 安装依赖包
```bash
pip install -r requirements.txt
```

## 使用方法

### 运行主程序

```bash
python main.py
```

### 参数说明

在 [main.py](main.py) 中可以调整以下参数：

- `num_samples`: 测试样本数量（默认100）
- `max_swaps_list`: 最大词替换数量列表（默认[1, 2, 3, 4, 5]）
- `device`: 运行设备（自动检测CUDA）

## 核心模块说明

### 1. 数据集准备 ([dataset_prep.py](dataset_prep.py))

**load_model_and_tokenizer()**
- 加载预训练的BERT分类模型
- 模型：textattack/bert-base-uncased-rotten-tomatoes
- 支持自定义模型路径

**load_test_data()**
- 加载Rotten Tomatoes电影评论数据集
- 支持指定样本数量
- 自动缓存数据集到本地

### 2. 攻击生成 ([attack_gen.py](attack_gen.py))

**get_word_saliency()**
- 计算每个token的梯度显著性
- 使用L2范数作为显著性分数
- 忽略特殊token（[CLS], [SEP], [PAD]）

**gradient_based_word_swap()**
- 执行基于梯度的词替换攻击
- 将显著性最高的N个词替换为[UNK]
- 支持自定义替换标记和最大替换数量

### 3. 评估指标 ([evaluation.py](evaluation.py))

**predict_text()**
- 对单条文本进行预测
- 返回预测标签

**calculate_word_change_rate()**
- 计算词汇级别的修改率
- 作为感知相似性的替代指标

**highlight_adversarial_text()**
- 在终端中高亮显示对抗文本的修改部分
- 使用ANSI颜色代码突出显示[UNK]标记

### 4. 可视化工具 ([visualization.py](visualization.py))

**plot_nlp_tradeoff_curves()**
- 生成NLP对抗攻击的量化评估曲线
- 包含扰动强度vs ASR和词汇修改率vs ASR两个子图

**plot_perceptual_tradeoff()**
- 绘制ASR与感知相似性的权衡曲线
- 支持自定义指标名称

**generate_html_report()**
- 生成直观的HTML报告
- 展示攻击前后的语句变化
- 高亮显示被替换的token

## 实验结果

程序运行后会在 `results/` 目录下生成以下文件：

1. **nlp_tradeoff_curves.png**: NLP评估曲线
   - 攻击强度（最大词替换数）vs 攻击成功率
   - 词汇修改率 vs 攻击成功率

2. **perceptual_tradeoff_WordChangeRate_as_Perceptual.png**: 感知权衡曲线
   - 展示攻击成功率与文本修改程度的权衡关系

3. **adversarial_examples_report.html**: 对抗样本HTML报告
   - 展示成功攻击的案例
   - 对比原始文本和对抗文本
   - 高亮显示被替换的词汇

## 评估指标

### 攻击成功率 (ASR)
```
ASR = 成功攻击数量 / 原本预测正确的样本数量
```

### 词汇修改率
```
Word Change Rate = 改变的词汇数量 / 原始词汇总数
```

## 技术细节

### 攻击原理

1. **梯度计算**：通过反向传播计算损失函数对输入embedding的梯度
2. **显著性分析**：计算每个token梯度的L2范数作为显著性分数
3. **目标选择**：选择显著性分数最高的N个token进行替换
4. **扰动生成**：将选中的token替换为[UNK]标记

### 模型架构

- **基础模型**: BERT-base (uncased)
- **任务**: 二分类（正面/负面情感）
- **训练数据**: Rotten Tomatoes电影评论

## 示例输出

```
[*] 使用设备: cuda
[*] 正在加载模型和 Tokenizer: textattack/bert-base-uncased-rotten-tomatoes
[*] 正在加载数据集 (取 100 条测试)...

------------------------------------------------------------
[-] 正在测试攻击强度: 替换最多 1 个梯度显著词
Attacking (swaps=1): 100%|████████████| 100/100 [00:05<00:00, 18.5it/s]
    -> [基线] 模型原始准确率: 85.00%
    -> [攻击] 攻击成功率 (ASR): 12.94%
    -> [相似性] 平均词汇修改率: 3.21%

[+] 成功案例展示 (Max Swaps: 1):
    原始文本 (标签 1): This movie is absolutely wonderful!
    对抗文本 (标签 0): This movie is absolutely [UNK]!
```