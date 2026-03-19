import torch
import torch.nn as nn
import torch.optim as optim
import random
import config
from torchmetrics.image import StructuralSimilarityIndexMeasure # 引入 SSIM

class PatchAttacker:
    def __init__(self, model, device, patch_size):
        self.model = model
        self.device = device
        self.patch_size = patch_size # 动态接收补丁尺寸
        self.patch = torch.rand((3, self.patch_size, self.patch_size), device=self.device, requires_grad=True)
        self.optimizer = optim.Adam([self.patch], lr=config.LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()
        
        # 初始化 SSIM 计算器，数据范围为 [0, 1]
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)

    def apply_patch(self, images):
        patched_images = images.clone()
        b, c, h, w = patched_images.shape
        for i in range(b):
            start_x = random.randint(0, w - self.patch_size - 1)
            start_y = random.randint(0, h - self.patch_size - 1)
            patched_images[i, :, start_y:start_y+self.patch_size, start_x:start_x+self.patch_size] = self.patch
        return patched_images

    def train_patch(self, dataloader):
        print(f"\n[>>>] 开始训练补丁，当前尺寸: {self.patch_size}x{self.patch_size}")
        for epoch in range(config.EPOCHS):
            running_loss = 0.0
            for images, labels in dataloader:
                images = images.to(self.device)
                mask = labels != config.TARGET_CLASS
                if not mask.any(): continue
                images = images[mask]
                target_labels = torch.full((images.size(0),), config.TARGET_CLASS, dtype=torch.long, device=self.device)
                
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

    def evaluate(self, dataloader):
        """测试生成的补丁在测试集上的效果，并计算 ASR 和 SSIM"""
        success_count = 0
        total = 0
        total_ssim = 0.0
        batch_count = 0
        
        print(f"[*] 正在评估尺寸 {self.patch_size}x{self.patch_size} 的补丁...")
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                mask = labels != config.TARGET_CLASS
                if not mask.any(): continue
                images, labels = images[mask], labels[mask]
                target_labels = torch.full((images.size(0),), config.TARGET_CLASS, dtype=torch.long, device=self.device)
                
                patched_images = self.apply_patch(images)
                outputs = self.model(patched_images)
                _, predicted = outputs.max(1)
                
                total += target_labels.size(0)
                success_count += predicted.eq(target_labels).sum().item()
                
                # 计算这一个 batch 的原图与被攻击图的 SSIM
                batch_ssim = self.ssim_metric(patched_images, images)
                total_ssim += batch_ssim.item()
                batch_count += 1
                
        asr = 100. * success_count / total
        avg_ssim = total_ssim / batch_count
        print(f"[*] 尺寸 {self.patch_size}x{self.patch_size} | ASR: {asr:.2f}% | 平均 SSIM: {avg_ssim:.4f}")
        return asr, avg_ssim