import time
import torch
import config

def evaluate_pipeline(model, dataloader, attacker, detector):
    clean_correct = 0
    adv_correct = 0
    defended_clean_correct = 0
    defended_adv_correct = 0
    total = 0
    
    total_clean_time = 0
    total_defend_time = 0

    results = {'clean_images': [], 'adv_images': [], 'labels': [], 'clean_preds': [], 'adv_preds': []}

    # ==========================================
    # 阶段一：防御器校准 (Calibration)
    # ==========================================
    # 获取第一批数据用于计算阈值
    calib_images, calib_labels = next(iter(dataloader))
    calib_images, calib_labels = calib_images.to(config.DEVICE), calib_labels.to(config.DEVICE)
    
    # 仅使用模型本来就能预测正确的干净样本来校准
    model.eval()
    calib_outputs = model(calib_images)
    _, calib_preds = torch.max(calib_outputs, 1)
    correct_mask = (calib_preds == calib_labels)
    
    # 执行校准（设定 95% 置信区间）
    detector.calibrate(calib_images[correct_mask], calib_labels[correct_mask], percentile=80)

    # ==========================================
    # 阶段二：量化评估 (Evaluation)
    # ==========================================
    print("[*] 开始全量数据评估...")
    for images, labels in dataloader:
        images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
        total += labels.size(0)

        # 1. 干净样本原始推理与耗时计算
        start_time = time.time()
        clean_outputs = model(images)
        _, clean_preds = torch.max(clean_outputs.data, 1)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_clean_time += time.time() - start_time
        clean_correct += (clean_preds == labels).sum().item()

        # 2. 生成对抗样本
        adv_images = attacker(images, labels)
        adv_outputs = model(adv_images)
        _, adv_preds = torch.max(adv_outputs.data, 1)
        adv_correct += (adv_preds == labels).sum().item()

        # 3. 干净样本的防御评估与延迟计算
        start_time_def = time.time()
        clean_anomalies, _ = detector.detect(images, clean_preds)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        total_defend_time += time.time() - start_time_def
        
        # 干净样本精度 = 本来预测正确 & 且没有被误判为异常
        defended_clean_correct += ((clean_preds == labels) & (~clean_anomalies)).sum().item()

        # 4. 对抗样本的防御评估
        adv_anomalies, _ = detector.detect(adv_images, adv_preds)
        # 防御成功 = 成功识别出对抗样本 (异常) OR 模型依然预测正确
        defended_adv_correct += (adv_anomalies | (adv_preds == labels)).sum().item()

        # 收集第一批的图片用于可视化
        if len(results['clean_images']) == 0:
            results['clean_images'] = images.cpu()
            results['adv_images'] = adv_images.cpu()
            results['labels'] = labels.cpu()
            results['clean_preds'] = clean_preds.cpu()
            results['adv_preds'] = adv_preds.cpu()
            results['adv_anomalies'] = adv_anomalies.cpu()

    metrics = {
        'clean_acc': clean_correct / total,
        'adv_acc': adv_correct / total,
        'defended_clean_acc': defended_clean_correct / total,
        'defended_adv_acc': defended_adv_correct / total,
        'avg_clean_latency': total_clean_time / total,
        'avg_defend_latency': total_defend_time / total,
    }
    return metrics, results