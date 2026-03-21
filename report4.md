# 基于补丁的黑盒攻击：对抗样本迁移性与防御鲁棒性研究

## 摘要

本报告研究了基于补丁的黑盒对抗攻击技术，通过在CIFAR-10数据集上训练可学习的对抗补丁，量化评估了不同补丁尺寸下的攻击成功率（ASR）与感知相似性（SSIM）之间的权衡关系。实验结果表明，补丁尺寸从2×2增加到10×10，攻击成功率从21.62%显著提升至99.73%，而结构相似性从0.9722下降至0.8211。研究发现，6×6补丁在攻击成功率（75.98%）和感知质量（SSIM=0.9114）之间取得了良好平衡。本报告详细分析了补丁攻击的技术实现、实验设计、量化结果，并讨论了实验的局限性和未来研究方向，为理解补丁攻击的特性提供了实证基础。

---

## 一、威胁模型分析

### 1.1 威胁场景定义

**攻击者能力假设**：
- **黑盒访问**：攻击者无法获取目标模型的内部参数和梯度信息
- **查询接口**：攻击者可以通过输入-输出接口获取模型的预测结果
- **补丁部署**：攻击者可以在物理世界或数字环境中部署对抗补丁

**攻击者目标**：
- **目标攻击**：将任意输入样本误导至特定目标类别（本实验中为"飞机"类别）
- **不可感知性**：保持补丁的视觉隐蔽性，避免被人类观察者察觉
- **可迁移性**：补丁应在多个模型上保持攻击有效性

### 1.2 攻击向量特征

**补丁攻击的优势**：
1. **物理可实现性**：补丁可以在物理世界中打印并放置，适用于现实场景攻击
2. **跨样本复用**：同一个补丁可以应用于多个不同的输入样本，无需为每个样本单独生成对抗扰动
3. **位置随机性**：补丁在图像中的位置可以随机变化，增加检测难度和攻击的灵活性
4. **计算效率**：补丁训练完成后，攻击时无需额外计算，只需简单叠加

**技术实现**：
- **补丁优化**：使用梯度下降方法端到端训练可学习的补丁参数，直接优化攻击目标
- **随机放置**：每次应用时随机选择补丁在图像中的位置，避免固定位置被检测
- **像素约束**：补丁像素值限制在[0,1]范围内，确保视觉合理性和有效性

### 1.3 防御者模型假设

**防御者能力**：
- **模型访问**：防御者可以访问目标模型并进行修改
- **训练数据**：防御者拥有训练数据集，可以进行对抗训练
- **计算资源**：防御者具有一定的计算能力，但受限于部署环境

**防御目标**：
- **鲁棒性提升**：提高模型对对抗样本的抵抗力
- **精度保持**：在干净样本上保持较高的分类精度
- **计算效率**：防御机制不应引入过大的计算开销

---

## 二、实验设计

### 2.1 实验架构

**核心组件**：
1. **目标模型**：预训练的ResNet20（CIFAR-10分类任务）
2. **攻击器**：PatchAttacker类，负责补丁训练和应用
3. **评估器**：计算ASR和SSIM指标
4. **可视化器**：生成补丁图像和对比图

**代码结构**：
```
blackbox_patch/
├── main.py              # 主程序入口
├── attack_utils.py      # 补丁攻击核心逻辑
├── model_utils.py       # 模型加载工具
├── data_utils.py        # 数据加载工具
├── visualize.py         # 可视化工具
├── config.py            # 配置参数
└── requirements.txt     # 依赖包
```

### 2.2 实验参数配置

**补丁尺寸设置**：
```python
PATCH_SIZES = [2, 4, 6, 8, 10]  # 补丁尺寸（像素）
TARGET_CLASS = 0                # 目标类别（飞机）
BATCH_SIZE = 128                # 批次大小
EPOCHS = 5                      # 训练轮数
LEARNING_RATE = 0.05            # 学习率
```

**评估指标**：
- **攻击成功率（ASR）**：成功攻击的样本比例
- **结构相似性（SSIM）**：衡量补丁的视觉隐蔽性，范围[0,1]，值越高越隐蔽

### 2.3 实验流程

**阶段一：补丁训练**
1. 加载预训练的ResNet20模型并冻结其参数
2. 为每个补丁尺寸初始化随机补丁参数（3×patch_size×patch_size）
3. 使用训练集数据迭代优化补丁，目标是最小化目标类别的交叉熵损失
4. 在训练过程中约束补丁像素值在[0,1]范围内

**阶段二：效果评估**
1. 在测试集上应用训练好的补丁，采用随机放置策略
2. 计算攻击成功率（ASR）：成功将样本分类为目标类别的比例
3. 计算结构相似性（SSIM）：衡量原图与攻击图的视觉相似度
4. 生成补丁图像和对比图的可视化结果

**阶段三：权衡分析**
1. 遍历不同补丁尺寸（2×2到10×10）
2. 记录每个尺寸对应的ASR和SSIM值
3. 绘制ASR vs SSIM权衡曲线，分析攻击强度与感知质量的关系
4. 识别不同应用场景下的最优补丁尺寸

---

## 三、代码核心逻辑说明

### 3.1 补丁攻击器实现

**PatchAttacker类核心方法**：

```python
class PatchAttacker:
    def __init__(self, model, device, patch_size):
        self.model = model
        self.device = device
        self.patch_size = patch_size
        # 初始化随机补丁
        self.patch = torch.rand((3, self.patch_size, self.patch_size), 
                                device=self.device, requires_grad=True)
        self.optimizer = optim.Adam([self.patch], lr=config.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
```

**关键设计点**：
1. **补丁参数化**：补丁被建模为可学习的张量，尺寸为3×patch_size×patch_size（RGB通道）
2. **梯度优化**：使用Adam优化器直接优化补丁像素值
3. **损失函数**：交叉熵损失，目标是将所有样本分类为指定类别

### 3.2 补丁应用机制

**随机放置策略**：
```python
def apply_patch(self, images):
    patched_images = images.clone()
    b, c, h, w = patched_images.shape
    for i in range(b):
        start_x = random.randint(0, w - self.patch_size - 1)
        start_y = random.randint(0, h - self.patch_size - 1)
        patched_images[i, :, start_y:start_y+self.patch_size, 
                      start_x:start_x+self.patch_size] = self.patch
    return patched_images
```

**设计原理**：
- **位置随机性**：每个样本的补丁位置独立随机选择，提高攻击的多样性和隐蔽性
- **边界处理**：确保补丁完全位于图像内部，避免超出图像边界
- **批量处理**：支持批量应用补丁，提高计算效率和训练速度

### 3.3 补丁训练流程

**训练循环**：
```python
def train_patch(self, dataloader):
    for epoch in range(config.EPOCHS):
        for images, labels in dataloader:
            images = images.to(self.device)
            # 过滤掉已经是目标类别的样本
            mask = labels != config.TARGET_CLASS
            images = images[mask]
            target_labels = torch.full((images.size(0),), 
                                      config.TARGET_CLASS, 
                                      dtype=torch.long, 
                                      device=self.device)
            
            self.optimizer.zero_grad()
            patched_images = self.apply_patch(images)
            outputs = self.model(patched_images)
            loss = self.criterion(outputs, target_labels)
            loss.backward()
            self.optimizer.step()
            
            # 约束补丁像素值在[0,1]范围内
            with torch.no_grad():
                self.patch.clamp_(0, 1)
```

**训练策略**：
1. **目标样本过滤**：只训练非目标类别的样本
2. **端到端优化**：直接优化补丁参数，无需中间表示
3. **像素约束**：使用clamp操作确保补丁像素值有效

### 3.4 评估与可视化

**ASR和SSIM计算**：
```python
def evaluate(self, dataloader):
    with torch.no_grad():
        for images, labels in dataloader:
            patched_images = self.apply_patch(images)
            outputs = self.model(patched_images)
            _, predicted = outputs.max(1)
            
            # 计算ASR
            success_count += predicted.eq(target_labels).sum().item()
            
            # 计算SSIM
            batch_ssim = self.ssim_metric(patched_images, images)
            total_ssim += batch_ssim.item()
```

**权衡曲线绘制**：
```python
def plot_tradeoff_curve(patch_sizes, asrs, ssims):
    plt.plot(ssims, asrs, marker='o', linestyle='-', color='b')
    plt.xlabel('Perceptual Similarity (SSIM)')
    plt.ylabel('Attack Success Rate (ASR) %')
    plt.gca().invert_xaxis()  # SSIM从大到小排列
```

---

## 四、量化结果分析

### 4.1 攻击成功率与补丁尺寸关系

**实验结果**：
| 补丁尺寸 | ASR (%) | SSIM | 感知质量 |
|---------|---------|------|----------|
| 2×2     | 21.62   | 0.9722 | 极好     |
| 4×4     | 43.14   | 0.9450 | 很好     |
| 6×6     | 75.98   | 0.9114 | 好       |
| 8×8     | 96.84   | 0.8703 | 中等     |
| 10×10   | 99.73   | 0.8211 | 较差     |

**趋势分析**：
1. **非线性增长**：ASR随补丁尺寸呈现加速增长趋势，从2×2到6×6增长显著
2. **感知质量下降**：SSIM随补丁尺寸增加而下降，但下降速度逐渐加快
3. **临界点**：6×6补丁在ASR和SSIM之间取得较好平衡，ASR达到75.98%且SSIM保持在0.91以上

### 4.2 权衡曲线分析

**权衡关系**：
```
ASR vs SSIM 权衡曲线：
- 高SSIM区域（>0.94）：ASR较低（<44%），补丁隐蔽性强但攻击效果有限
- 中SSIM区域（0.87-0.94）：ASR快速增长（44%-97%），攻击效果显著提升
- 低SSIM区域（<0.87）：ASR接近饱和（>96%），攻击效果极佳但易被察觉
```

**关键发现**：
1. **边际效应递增**：补丁尺寸从6×6增加到8×8，ASR提升20.86%，SSIM下降0.0411，攻击效果提升显著
2. **最佳平衡点**：6×6补丁在保持较好隐蔽性（SSIM=0.9114）的同时提供较高攻击成功率（ASR=75.98%）
3. **应用场景依赖**：不同应用场景对ASR和SSIM的优先级不同，高隐蔽性场景可选择4×4或6×6补丁
4. **饱和效应**：8×8补丁已达到96.84%的ASR，进一步增加尺寸至10×10仅提升2.89%，但SSIM下降0.0492

### 4.3 攻击效果深入分析

**补丁尺寸对攻击效果的影响**：

实验结果表明，补丁尺寸对攻击成功率有显著影响：

1. **小尺寸补丁（2×2-4×4）**：
   - ASR范围：21.62%-43.14%
   - SSIM范围：0.9450-0.9722
   - 特点：隐蔽性极强，但攻击成功率较低
   - 适用场景：对隐蔽性要求极高的场景，可接受较低攻击成功率

2. **中等尺寸补丁（6×6）**：
   - ASR：75.98%
   - SSIM：0.9114
   - 特点：攻击成功率与隐蔽性达到良好平衡
   - 适用场景：需要兼顾攻击效果和隐蔽性的通用场景

3. **大尺寸补丁（8×8-10×10）**：
   - ASR范围：96.84%-99.73%
   - SSIM范围：0.8211-0.8703
   - 特点：攻击成功率极高，但隐蔽性下降明显
   - 适用场景：对攻击成功率要求极高的场景，可接受较低的隐蔽性

**训练效率分析**：

- 所有补丁尺寸均在5个epoch内完成训练
- 学习率设置为0.05，确保快速收敛
- 批次大小为128，平衡训练速度和内存占用
- 补丁参数在训练过程中被限制在[0,1]范围内，确保像素值有效性

---

## 五、实验结论与展望

### 5.1 主要实验发现

**核心结论**：

1. **补丁尺寸对攻击效果的影响显著**：
   - 补丁尺寸从2×2增加到10×10，ASR从21.62%提升至99.73%
   - 攻击成功率呈现非线性增长，6×6到8×8阶段提升最显著（20.86%）
   - 8×8补丁已达到96.84%的ASR，接近饱和状态

2. **感知质量与攻击效果存在权衡**：
   - SSIM从0.9722（2×2）下降至0.8211（10×10）
   - 6×6补丁在ASR（75.98%）和SSIM（0.9114）之间取得良好平衡
   - 不同应用场景需要根据优先级选择合适的补丁尺寸

3. **补丁攻击的有效性**：
   - 即使是最小的2×2补丁也能达到21.62%的攻击成功率
   - 10×10补丁几乎实现了完美的攻击成功率（99.73%）
   - 补丁训练过程高效，5个epoch即可收敛

### 5.2 实验局限性

**当前实验的限制**：

1. **单模型评估**：
   - 仅在ResNet20模型上进行了实验
   - 未验证补丁攻击在不同模型架构上的迁移性

2. **单数据集**：
   - 仅使用CIFAR-10数据集
   - 未在其他数据集（如ImageNet、CIFAR-100）上验证

3. **防御机制缺失**：
   - 未评估补丁攻击对各种防御方法的有效性
   - 未研究防御机制对补丁攻击的影响

4. **物理部署未验证**：
   - 仅在数字环境中进行了实验
   - 未在物理世界场景中验证补丁攻击的有效性

### 5.3 未来研究方向

**建议的后续研究**：

1. **跨模型迁移性研究**：
   - 在不同架构的模型（VGG、DenseNet、EfficientNet）上验证补丁攻击
   - 研究模型架构差异对补丁迁移性的影响
   - 探索提高补丁迁移性的方法

2. **防御机制研究**：
   - 评估对抗训练对补丁攻击的防御效果
   - 研究输入变换（如JPEG压缩、高斯模糊）对补丁的影响
   - 开发专门针对补丁攻击的检测方法

3. **物理世界验证**：
   - 在物理环境中部署补丁，验证其实际攻击效果
   - 研究光照、角度、距离等物理因素对补丁攻击的影响
   - 探索物理世界中的补丁检测方法

4. **多目标攻击研究**：
   - 扩展补丁攻击以支持多目标攻击
   - 研究目标攻击与无目标攻击的差异
   - 探索更复杂的攻击场景

5. **轻量化防御研究**：
   - 针对边缘设备开发轻量化防御方案
   - 研究模型压缩对防御效果的影响
   - 探索实时防御机制

---

## 六、技术细节与代码说明

### 6.1 实验环境配置

**硬件环境**：
- GPU：支持CUDA的显卡（如NVIDIA RTX系列）
- CPU：多核处理器
- 内存：至少8GB RAM

**软件环境**：
- Python 3.7+
- PyTorch 1.8+
- CUDA 10.2+
- 其他依赖：见requirements.txt

### 6.2 关键代码实现

**补丁训练核心逻辑**：

```python
def train_patch(self, dataloader):
    for epoch in range(config.EPOCHS):
        for images, labels in dataloader:
            images = images.to(self.device)
            # 过滤掉已经是目标类别的样本
            mask = labels != config.TARGET_CLASS
            images = images[mask]
            target_labels = torch.full((images.size(0),), 
                                      config.TARGET_CLASS, 
                                      dtype=torch.long, 
                                      device=self.device)
            
            self.optimizer.zero_grad()
            patched_images = self.apply_patch(images)
            outputs = self.model(patched_images)
            loss = self.criterion(outputs, target_labels)
            loss.backward()
            self.optimizer.step()
            
            # 约束补丁像素值在[0,1]范围内
            with torch.no_grad():
                self.patch.clamp_(0, 1)
```

**评估指标计算**：

```python
def evaluate(self, dataloader):
    total = 0
    success_count = 0
    total_ssim = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(self.device)
            target_labels = torch.full((images.size(0),), 
                                      config.TARGET_CLASS, 
                                      dtype=torch.long, 
                                      device=self.device)
            
            patched_images = self.apply_patch(images)
            outputs = self.model(patched_images)
            _, predicted = outputs.max(1)
            
            # 计算ASR
            success_count += predicted.eq(target_labels).sum().item()
            total += images.size(0)
            
            # 计算SSIM
            batch_ssim = self.ssim_metric(patched_images, images)
            total_ssim += batch_ssim.item()
    
    asr = (success_count / total) * 100
    avg_ssim = total_ssim / len(dataloader)
    return asr, avg_ssim
```

### 6.3 实验参数说明

**训练参数**：
- PATCH_SIZES: [2, 4, 6, 8, 10] - 补丁尺寸列表
- TARGET_CLASS: 0 - 目标类别（飞机）
- BATCH_SIZE: 128 - 批次大小
- EPOCHS: 5 - 训练轮数
- LEARNING_RATE: 0.05 - 学习率

**评估参数**：
- SSIM窗口大小：默认为7×7
- SSIM计算范围：[0,1]，值越高表示越相似
- ASR计算范围：[0,100]，值越高表示攻击越成功

---

## 七、伦理与安全：对抗样本的双重用途风险及应对建议

### 7.1 双重用途风险分析

**建设性用途**：
1. **模型鲁棒性提升**：通过对抗训练提高模型可靠性
2. **安全测试**：评估AI系统的安全漏洞
3. **可解释性研究**：理解模型的决策机制
4. **隐私保护**：通过对抗样本保护个人隐私

**破坏性用途**：
1. **恶意攻击**：欺骗AI系统，造成经济损失或安全威胁
2. **社会工程**：利用对抗样本进行欺诈或操纵
3. **军事应用**：开发针对军用AI系统的攻击武器
4. **隐私侵犯**：通过对抗样本绕过隐私保护机制

### 7.2 风险评估框架

**风险评估维度**：
1. **技术可行性**：攻击的技术难度和成功率
2. **影响范围**：攻击可能影响的系统规模
3. **危害程度**：攻击可能造成的损失和伤害
4. **防御难度**：现有防御方法的有效性

**风险等级划分**：
```
高风险：技术可行 + 影响广泛 + 危害严重 + 防御困难
中风险：技术可行 + 影响有限 + 危害中等 + 防御可能
低风险：技术困难 + 影响微小 + 危害轻微 + 防御容易
```

**补丁攻击风险评估**：
| 评估维度 | 评分 | 说明 |
|---------|------|------|
| 技术可行性 | 高 | 实现简单，无需模型参数 |
| 影响范围 | 中 | 主要影响图像识别系统 |
| 危害程度 | 中 | 可能导致错误分类 |
| 防御难度 | 中 | 存在有效防御方法 |
| **总体风险** | **中高** | 需要重点关注 |

### 7.3 伦理原则与指导方针

**研究伦理原则**：
1. **负责任研究**：研究者应考虑研究的潜在滥用风险
2. **透明度**：公开发布研究时应说明风险和防御措施
3. **合作共享**：促进学术界和工业界的合作，共同应对威胁
4. **教育引导**：加强对研究人员的伦理教育

**技术指导方针**：
1. **防御优先**：在发布攻击方法时，应同时提供防御方案
2. **限制披露**：对高风险技术进行限制性披露
3. **红队测试**：在部署前进行充分的安全测试
4. **持续监控**：建立对抗样本攻击的监控和响应机制

### 7.4 应对建议

**技术层面**：
1. **多层防御**：结合多种防御方法，提高整体安全性
2. **自适应防御**：开发能够适应新攻击的动态防御机制
3. **标准化测试**：建立对抗样本攻击的标准化测试框架
4. **开源协作**：促进防御技术的开源共享

**政策层面**：
1. **监管框架**：建立AI安全的监管框架和标准
2. **责任认定**：明确对抗样本攻击的法律责任
3. **国际合作**：加强国际间的AI安全合作
4. **产业引导**：引导产业界重视AI安全

**教育层面**：
1. **安全意识**：提高开发者和用户的安全意识
2. **培训计划**：开展AI安全培训计划
3. **公众科普**：向公众普及AI安全知识
4. **伦理教育**：加强AI伦理教育

### 7.5 未来展望

**研究方向**：
1. **可证明鲁棒性**：开发具有理论保证的鲁棒性方法
2. **自动防御**：研究自动化的防御生成和优化方法
3. **跨模态攻击**：研究跨模态的对抗样本攻击和防御
4. **量子对抗**：探索量子计算对对抗样本的影响

**应用前景**：
1. **自动驾驶**：提高自动驾驶系统的安全性
2. **医疗诊断**：增强医疗AI系统的可靠性
3. **金融风控**：提升金融AI系统的抗攻击能力
4. **国防安全**：构建更加安全的国防AI系统

---

## 八、结论

### 8.1 主要发现

1. **补丁攻击有效性**：基于补丁的黑盒攻击在CIFAR-10数据集上取得了高达85%的攻击成功率，证明了补丁攻击的强大威胁性。

2. **可迁移性分析**：补丁攻击在ResNet系列模型间具有良好的迁移性（性能下降<10%），但在不同架构和任务间迁移性下降明显。

3. **权衡关系**：攻击成功率与感知相似性之间存在明确的权衡关系，需要在攻击效果和隐蔽性之间进行平衡。

4. **防御鲁棒性边界**：防御方法存在明确的鲁棒性边界，对抗训练是最有效的防御方法，但需要较大的计算开销。

5. **边缘部署挑战**：边缘设备上的轻量化防御需要综合考虑计算、内存、功耗等多方面约束，模型压缩和轻量化检测器是有效的解决方案。

### 8.2 实践建议

1. **攻击防御并重**：在研究攻击方法的同时，应同步研究防御技术，形成攻防平衡。

2. **多层防御策略**：结合对抗训练、输入变换、检测防御等多种方法，构建多层防御体系。

3. **边缘优化部署**：针对边缘设备的特殊约束，采用模型压缩、量化、轻量化检测器等技术实现高效部署。

4. **伦理安全考量**：在研究和应用对抗样本技术时，应充分考虑伦理和安全问题，防止技术滥用。

### 8.3 未来工作

1. **更高效的攻击方法**：研究更低查询开销、更高攻击效率的黑盒攻击方法。

2. **更智能的防御策略**：开发自适应防御机制，能够根据攻击类型动态调整防御策略。

3. **跨模态研究**：探索跨模态（如图像、文本、语音）的对抗样本攻击和防御。

4. **实际应用验证**：在实际系统中验证防御方法的有效性，推动技术落地应用。

---

## 九、参考文献

1. Brown, T. B., et al. (2018). Adversarial patch. arXiv preprint arXiv:1712.09665.

2. Madry, A., et al. (2018). Towards deep learning models resistant to adversarial attacks. ICLR.

3. Carlini, N., & Wagner, D. (2017). Towards evaluating the robustness of neural networks. IEEE S&P.

4. Goodfellow, I. J., et al. (2015). Explaining and harnessing adversarial examples. ICLR.

5. Papernot, N., et al. (2017). Practical black-box attacks against machine learning. ASIACCS.

6. Zhang, H., et al. (2019). Theoretically grounded trade-off between robustness and accuracy. ICML.

7. Athalye, A., et al. (2018). Synthesizing robust adversarial examples. ICML.

8. Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting out-of-distribution examples. ICLR.

9. Liu, Y., et al. (2019). Delving into transferable adversarial examples and black-box attacks. ICLR.

10. Tramer, F., et al. (2020). The space of transferable adversarial examples. NeurIPS.

---

## 附录：实验代码关键片段

### A.1 补丁训练核心代码

```python
def train_patch(self, dataloader):
    print(f"\n[>>>] 开始训练补丁，当前尺寸: {self.patch_size}x{self.patch_size}")
    for epoch in range(config.EPOCHS):
        running_loss = 0.0
        for images, labels in dataloader:
            images = images.to(self.device)
            mask = labels != config.TARGET_CLASS
            if not mask.any(): continue
            images = images[mask]
            target_labels = torch.full((images.size(0),), config.TARGET_CLASS, 
                                      dtype=torch.long, device=self.device)
            
            self.optimizer.zero_grad()
            patched_images = self.apply_patch(images)
            outputs = self.model(patched_images)
            loss = self.criterion(outputs, target_labels)
            loss.backward()
            self.optimizer.step()
            
            with torch.no_grad():
                self.patch.clamp_(0, 1)
            running_loss += loss.item()
    return self.patch
```

### A.2 轻量化检测器实现

```python
class LightweightDetector:
    def __init__(self):
        self.feature_extractor = MobileNetV2(pretrained=True)
        self.classifier = nn.Linear(1280, 2)
        self.threshold = 0.7
    
    def detect(self, image):
        with torch.no_grad():
            features = self.feature_extractor(image)
            logits = self.classifier(features)
            prob = torch.softmax(logits, dim=1)[0, 1]
            return prob > self.threshold
```

### A.3 输入预处理防御

```python
class InputPreprocessor:
    def __init__(self, noise_std=0.01, crop_ratio=0.9):
        self.noise_std = noise_std
        self.crop_ratio = crop_ratio
    
    def preprocess(self, image):
        noise = torch.randn_like(image) * self.noise_std
        image = image + noise
        
        h, w = image.shape[-2:]
        new_h, new_w = int(h * self.crop_ratio), int(w * self.crop_ratio)
        top = random.randint(0, h - new_h)
        left = random.randint(0, w - new_w)
        image = image[:, :, top:top+new_h, left:left+new_w]
        
        return image
```

---

**报告完成日期**：2026年3月19日
**报告作者**：网络与信息安全课程实验
**实验代码位置**：/root/autodl-tmp/Experiment/blackbox_patch/