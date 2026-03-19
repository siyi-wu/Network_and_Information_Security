# evaluate.py
import torch
import time
from attack import pgd_attack

def evaluate_robustness_and_latency(model, data_loader, device, is_defended=False, test_batches=4):
    """
    量化评估模型性能
    :param test_batches: 为了加快运行速度，默认只测试前 4 个 Batch (128张图)。
                         如果在最终实验报告中需要完整数据，请将其设为 None。
    """
    model.eval()
    correct_clean = 0
    correct_adv = 0
    total = 0
    total_latency = 0.0

    print(f"--- 正在评估模型 (防御开启: {is_defended}) ---")
    
    # ---------------------------------------------------------
    # 核心修正：GPU Warmup (预热)
    # 提前跑几个 Batch，让 CUDA 初始化显存和内核，避免把初始化时间算入延迟
    # ---------------------------------------------------------
    print("  -> 正在进行 GPU 预热...")
    dummy_loader = iter(data_loader)
    for _ in range(3):
        try:
            dummy_images, _ = next(dummy_loader)
            with torch.no_grad():
                _ = model(dummy_images.to(device))
        except StopIteration:
            break
    torch.cuda.synchronize() # 强制等待预热计算全部完成
    print("  -> 预热完成，开始正式测速与评估。")

    for batch_idx, (images, labels) in enumerate(data_loader):
        if test_batches is not None and batch_idx >= test_batches:
            break
            
        images, labels = images.to(device), labels.to(device)
        total += labels.size(0)

        # 1. 测量干净样本精度与单图推理延迟
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            outputs_clean = model(images)
        end_event.record()
        torch.cuda.synchronize()
        
        # 累加这一个 Batch 的总耗时
        total_latency += start_event.elapsed_time(end_event) 
        
        _, predicted_clean = outputs_clean.max(1)
        correct_clean += predicted_clean.eq(labels).sum().item()

        # 2. 生成对抗样本并测试鲁棒性
        adv_images = pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, iters=10, device=device)
        
        with torch.no_grad():
            outputs_adv = model(adv_images)
            _, predicted_adv = outputs_adv.max(1)
            correct_adv += predicted_adv.eq(labels).sum().item()

    clean_acc = 100. * correct_clean / total
    adv_acc = 100. * correct_adv / total
    # 计算单张图片的平均延迟 (总毫秒数 / 总图片数)
    avg_latency = total_latency / total 

    return clean_acc, adv_acc, avg_latency