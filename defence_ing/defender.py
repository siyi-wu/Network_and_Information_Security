import torch
from captum.attr import IntegratedGradients
import config

class IGDetector:
    def __init__(self, model):
        self.model = model
        self.ig = IntegratedGradients(model)
        self.threshold = None  # 动态阈值初始为空

    def compute_score(self, images, labels):
        # 计算特征归因，n_steps=20 可平衡检测精度与推理延迟
        attributions = self.ig.attribute(images, target=labels, n_steps=20)
        
        # 计算每个样本归因图绝对值的方差，作为异常分数
        scores = torch.var(torch.abs(attributions), dim=(1, 2, 3))
        return scores, attributions

    def calibrate(self, clean_images, clean_labels, percentile=95):
        """
        基于干净样本自动计算动态阈值
        percentile=95 表示允许 5% 的干净样本被误判为异常（即牺牲 5% 的干净精度）
        """
        print(f"[*] 正在使用 {len(clean_images)} 个干净样本校准检测器...")
        scores, _ = self.compute_score(clean_images, clean_labels)
        
        # 使用 torch.quantile 计算分位数
        self.threshold = torch.quantile(scores, percentile / 100.0).item()
        print(f"[*] 校准完成！基于 {percentile}% 置信区间的动态阈值为: {self.threshold:.6f}")

    def detect(self, images, labels):
        if self.threshold is None:
            raise ValueError("检测器尚未校准！请先调用 calibrate() 方法。")
            
        scores, attributions = self.compute_score(images, labels)
        
        # 大于动态阈值的视为异常样本 (对抗样本)
        is_anomaly = scores > self.threshold
        return is_anomaly, attributions