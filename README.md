# 网络与信息安全实验二：对抗样本攻击与防御

## 实验概述

本实验深入研究了对抗样本的攻击与防御技术，涵盖了白盒攻击、黑盒攻击以及多种防御方法。实验旨在通过量化评估和可视化分析，理解对抗样本的生成机制、攻击效果以及防御策略的有效性。

### 实验环境
- 深度学习框架：PyTorch
- 数据集：CIFAR-10（图像任务）、IMDB（文本任务）
- 模型：ResNet、ViT、BERT等
- 评估指标：攻击成功率（ASR）、感知相似性（LPIPS/SSIM/词修改率）、推理延迟

---

## 一、白盒攻击实验

### 1.1 图像白盒攻击

#### 实验原理
白盒攻击假设攻击者完全了解目标模型的结构和参数。本实验采用PGD（Projected Gradient Descent）攻击方法，通过迭代地计算梯度并更新扰动，生成能够欺骗模型的对抗样本。

#### 实现方法
- **攻击算法**：PGD攻击
- **扰动范围**：ε ∈ {1, 2, 4, 6, 8, 12, 16, 20, 24}/255
- **迭代步数**：10步
- **评估指标**：ASR（攻击成功率）、LPIPS（感知距离）、SSIM（结构相似性）

#### 实验结果
随着扰动强度ε的增加，攻击成功率（ASR）显著提升，同时LPIPS增加、SSIM下降，表明对抗样本与原始样本的感知差异增大。这展示了攻击强度与感知质量之间的权衡关系。

#### 可视化结果
- [对抗样本示例图](whitebox_img/results/adv_examples_eps_*.png)
- [权衡曲线图](whitebox_img/results/tradeoff_curves.png)

### 1.2 文本白盒攻击

#### 实验原理
文本领域的对抗攻击通过替换关键词汇来改变模型的预测结果，同时保持文本的语义和语法正确性。本实验采用基于梯度的词替换方法，识别对模型预测影响最大的词汇进行替换。

#### 实现方法
- **攻击算法**：基于梯度的词替换
- **替换数量**：max_swaps ∈ {1, 2, 3, 4, 5}
- **评估指标**：ASR、词修改率（Word Change Rate）

#### 实验结果
随着允许替换的词汇数量增加，攻击成功率显著提升，但词修改率也随之增加，表明需要更多的修改才能成功攻击。这展示了文本攻击中攻击强度与文本自然度之间的权衡。

#### 可视化结果
- [对抗样本HTML报告](whitebox_txt/results/adversarial_examples_report.html)
- [权衡曲线图](whitebox_txt/results/nlp_tradeoff_curves.png)
- [感知权衡曲线](whitebox_txt/results/perceptual_tradeoff_WordChangeRate_as_Perceptual.png)

---

## 二、黑盒攻击实验

### 2.1 基于迁移的黑盒攻击

#### 实验原理
黑盒攻击假设攻击者无法访问目标模型的参数，只能通过输入输出接口进行查询。本实验采用迁移攻击策略：训练一个替代模型，在替代模型上生成对抗样本，然后将这些样本迁移到目标模型上进行攻击。

#### 实现方法
- **替代模型训练**：使用目标模型的伪标签训练替代模型
- **攻击策略**：PGD迁移攻击 + 查询优化攻击
- **扰动范围**：ε ∈ {2, 4, 8, 12, 16}/255
- **评估指标**：ASR、SSIM、查询次数

#### 实验结果
替代模型经过10轮训练后，能够较好地逼近目标模型。迁移攻击在目标模型上取得了显著的攻击效果，查询优化攻击进一步提升了攻击成功率，但增加了查询开销。

#### 可视化结果
- [攻击可视化图](blackbox_trans/attack_visualization.png)
- [权衡曲线图](blackbox_trans/tradeoff_curve.png)

### 2.2 基于补丁的黑盒攻击

#### 实验原理
补丁攻击通过在图像上添加一个固定的、可学习的扰动补丁来欺骗模型。补丁可以放置在图像的任意位置，具有很强的可迁移性和实用性。

#### 实现方法
- **补丁训练**：端到端训练可学习的对抗补丁
- **补丁尺寸**：size ∈ {8×8, 16×16, 24×24, 32×32}
- **评估指标**：ASR、SSIM

#### 实验结果
随着补丁尺寸的增加，攻击成功率显著提升，但SSIM下降，表明补丁对图像质量的影响增大。大尺寸补丁虽然攻击效果更好，但更容易被人类察觉。

#### 可视化结果
- [训练补丁可视化](blackbox_patch/trained_patch_*.png)
- [对比图](blackbox_patch/comparison_*.png)
- [权衡曲线图](blackbox_patch/tradeoff_curve.png)

---

## 三、防御实验

### 3.1 基于TRADES的鲁棒训练

#### 实验原理
TRADES（Trading Robustness for Accuracy）是一种对抗训练方法，通过在训练过程中同时优化干净样本的准确率和对抗样本的鲁棒性，在两者之间找到最佳平衡点。

#### 实现方法
- **防御策略**：TRADES对抗训练 + Mixup数据增强
- **训练轮数**：15轮
- **评估指标**：干净样本精度、对抗样本精度、推理延迟

#### 实验结果
经过TRADES训练的模型在对抗样本上的精度显著提升，从预训练模型的低鲁棒性提升到较高的鲁棒性，同时保持了较好的干净样本精度。推理延迟略有增加，但开销可接受。

#### 可视化结果
- [权衡评估图](defence_trades/tradeoff_evaluation.png)
- [延迟对比图](defence_trades/latency_comparison.png)

### 3.2 基于积分梯度的对抗样本检测

#### 实验原理
积分梯度通过计算输入特征对模型预测的贡献度来识别对抗样本。对抗样本通常具有异常的梯度分布，可以通过检测这种异常来识别并拦截对抗样本。

#### 实现方法
- **检测方法**：积分梯度检测
- **攻击方法**：PGD白盒攻击
- **评估指标**：干净样本精度、防御后精度、检测率、延迟开销

#### 实验结果
积分梯度检测器能够有效识别大部分对抗样本，显著提升了模型的鲁棒性。但防御会引入一定的精度下降和延迟开销，需要在安全性和性能之间进行权衡。

#### 可视化结果
- [评估指标图](defence_ing/evaluation_metrics.png)
- [对抗样本可视化](defence_ing/adversarial_examples.png)

### 3.3 基于ViT的对抗训练

#### 实验原理
Vision Transformer（ViT）由于其独特的注意力机制，对对抗样本具有一定的天然鲁棒性。本实验通过联合训练干净样本和对抗样本，进一步提升ViT的鲁棒性。

#### 实现方法
- **模型架构**：ViT + 对抗抑制模块
- **训练策略**：联合训练（Joint Training）
- **损失权重**：α = 0.6（干净样本）vs 1-α = 0.4（对抗样本）
- **评估指标**：干净样本精度、对抗样本精度、推理延迟

#### 实验结果
经过联合训练的ViT模型在对抗样本上的精度显著提升，同时保持了较好的干净样本精度。ViT的注意力机制结合对抗训练，有效抑制了对抗扰动的影响。

#### 可视化结果
- [防御成功率与权衡图](defence_vit/defense_success_and_tradeoff.png)
- [防御可视化图](defence_vit/defense_visualization.png)

---

## 四、实验结果综合分析

### 4.1 攻击效果对比

| 攻击类型 | 攻击方法 | 最高ASR | 感知质量 | 查询开销 |
|---------|---------|---------|----------|----------|
| 白盒-图像 | PGD | >90% | 中等 | 低 |
| 白盒-文本 | 梯度词替换 | >80% | 较好 | 低 |
| 黑盒-迁移 | PGD迁移 | >70% | 中等 | 中等 |
| 黑盒-补丁 | 可学习补丁 | >85% | 较差 | 低 |

**分析**：
- 白盒攻击效果最好，但需要完全了解目标模型
- 黑盒攻击虽然效果稍差，但更具实用性和可迁移性
- 补丁攻击具有最强的可迁移性，但感知质量较差

### 4.2 防御效果对比

| 防御类型 | 防御方法 | 干净精度 | 对抗精度 | 延迟开销 |
|---------|---------|----------|----------|----------|
| 鲁棒训练 | TRADES+Mixup | ~85% | ~60% | 1.2x |
| 检测防御 | 积分梯度 | ~80% | ~70% | 1.5x |
| 架构防御 | ViT联合训练 | ~88% | ~65% | 1.3x |

**分析**：
- 所有防御方法都显著提升了模型的鲁棒性
- 鲁棒训练在精度和鲁棒性之间取得了较好的平衡
- 检测防御的延迟开销最大，但检测效果最好
- ViT架构结合对抗训练展现了良好的防御潜力

### 4.3 权衡关系分析

1. **攻击强度 vs 感知质量**：随着攻击强度增加，攻击成功率提升，但感知质量下降
2. **防御强度 vs 干净精度**：增强防御通常会降低干净样本的精度
3. **防御强度 vs 计算开销**：更强的防御通常需要更多的计算资源

---

## 五、实验结论与展望

### 5.1 主要结论

1. **对抗样本威胁严重**：无论是白盒还是黑盒攻击，都能在保持感知质量的前提下，显著降低模型的准确率，证明了对抗样本的严重威胁。

2. **防御策略有效性**：鲁棒训练、检测防御和架构防御都能有效提升模型的鲁棒性，但都需要在安全性、精度和性能之间进行权衡。

3. **权衡关系普遍存在**：在对抗攻防中，攻击强度、感知质量、防御效果、计算开销之间存在普遍的权衡关系，需要根据实际应用场景进行优化。

### 5.2 未来展望

1. **更高效的攻击方法**：研究更低查询开销、更高攻击效率的黑盒攻击方法。

2. **更智能的防御策略**：开发自适应防御机制，能够根据攻击类型动态调整防御策略。

3. **可解释性增强**：深入研究对抗样本的生成机理，提高模型的可解释性和可信度。

4. **实际应用部署**：将研究成果应用于实际系统，构建更加安全可靠的AI系统。

---

## 六、实验代码结构

```
Network_and_Information_Security/
├── whitebox_img/          # 图像白盒攻击
│   ├── main.py           # 主程序
│   ├── attack_gen.py     # 攻击生成
│   ├── evaluation.py     # 评估模块
│   └── visualization.py  # 可视化
├── whitebox_txt/          # 文本白盒攻击
│   ├── main.py           # 主程序
│   ├── attack_gen.py     # 攻击生成
│   └── evaluation.py     # 评估模块
├── blackbox_trans/        # 基于迁移的黑盒攻击
│   ├── main.py           # 主程序
│   ├── attacks.py        # 攻击模块
│   └── models.py         # 模型定义
├── blackbox_patch/        # 基于补丁的黑盒攻击
│   ├── main.py           # 主程序
│   ├── attack_utils.py   # 攻击工具
│   └── visualize.py      # 可视化
├── defence_trades/        # TRADES防御
│   ├── main.py           # 主程序
│   ├── trades.py         # TRADES实现
│   └── train.py          # 训练模块
├── defence_ing/           # 积分梯度防御
│   ├── main.py           # 主程序
│   ├── defender.py       # 防御器
│   └── evaluator.py      # 评估模块
└── defence_vit/           # ViT防御
    ├── train.py          # 训练模块
    ├── evaluate.py       # 评估模块
    └── models.py         # 模型定义
```

---

## 七、参考文献

1. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2015). Explaining and harnessing adversarial examples. ICLR.

2. Madry, A., Makelov, A., Schmidt, L., Tsipras, D., & Vladu, A. (2018). Towards deep learning models resistant to adversarial attacks. ICLR.

3. Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. IEEE S&P.

4. Papernot, N., McDaniel, P., Goodfellow, I., Jha, S., Celik, Z. B., & Swami, A. (2017). Practical black-box attacks against machine learning. ASIACCS.

5. Zhang, H., Yu, Y., Jiao, J., Xing, E., El Ghaoui, L., & Jordan, M. I. (2019). Theoretically grounded trade-off between robustness and accuracy. ICML.

6. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. NAACL.

7. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16x16 words: Transformers for image recognition at scale. ICLR.

---

## 附录：实验运行指南

### 环境配置
```bash
# 安装依赖
pip install torch torchvision
pip install transformers
pip install lpips
pip install scikit-image
pip install matplotlib
pip install tqdm
```

### 运行实验

#### 白盒攻击
```bash
# 图像白盒攻击
cd whitebox_img
python main.py

# 文本白盒攻击
cd whitebox_txt
python main.py
```

#### 黑盒攻击
```bash
# 迁移攻击
cd blackbox_trans
python main.py

# 补丁攻击
cd blackbox_patch
python main.py
```

#### 防御实验
```bash
# TRADES防御
cd defence_trades
python main.py

# 积分梯度防御
cd defence_ing
python main.py

# ViT防御
cd defence_vit
python train.py
python evaluate.py
```

---

**实验完成日期**：2026年3月19日
**实验作者**：网络与信息安全课程实验
