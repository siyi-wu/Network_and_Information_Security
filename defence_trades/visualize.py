import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_tradeoff(clean_acc_std, adv_acc_std, clean_acc_def, adv_acc_def):
    """绘制干净样本精度 vs 鲁棒性提升的权衡图"""
    labels = ['Standard Model', 'TRADES+Mixup Model']
    clean_accs = [clean_acc_std, clean_acc_def]
    adv_accs = [adv_acc_std, adv_acc_def]

    x = [1, 2]
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    rects1 = ax.bar([p - width/2 for p in x], clean_accs, width, label='Clean Accuracy', color='skyblue')
    rects2 = ax.bar([p + width/2 for p in x], adv_accs, width, label='Robust Accuracy (PGD)', color='salmon')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Defense Cost: Clean Accuracy Drop vs Robustness Improvement')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    # 添加数值标签
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.savefig('tradeoff_evaluation.png')
    print("Trade-off visualization saved as 'tradeoff_evaluation.png'")

def measure_inference_latency(model, device, input_shape=(1, 3, 32, 32), num_runs=100):
    """
    高精度测量模型在 GPU 上的单次推理延迟 (毫秒)
    """
    model.eval()
    dummy_input = torch.randn(input_shape).to(device)
    
    # 预热 (Warm-up) - 极其重要，否则第一次计算会包含初始化开销
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)
            
    # CUDA 计时器
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((num_runs, 1))
    
    with torch.no_grad():
        for i in range(num_runs):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # 同步 GPU 确保计算完成
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time
            
    # 计算平均延迟 (排除极端值)
    avg_latency = np.mean(timings)
    std_latency = np.std(timings)
    return avg_latency, std_latency

def plot_latency(latency_std, latency_rob):
    """绘制推理延迟对比柱状图"""
    labels = ['Standard Model', 'TRADES+Mixup Model']
    latencies = [latency_std, latency_rob]

    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(labels, latencies, color=['lightgray', 'lightgreen'], width=0.5)

    ax.set_ylabel('Inference Latency (ms / image)')
    ax.set_title('Computational Overhead: Inference Latency Comparison')
    
    # 设置 Y 轴从 0 开始，稍微留点余量
    ax.set_ylim(0, max(latencies) * 1.5)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f} ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

    plt.savefig('latency_comparison.png')
    print("Latency visualization saved as 'latency_comparison.png'")