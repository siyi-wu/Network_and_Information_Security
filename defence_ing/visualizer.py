import matplotlib.pyplot as plt
import numpy as np

def unnormalize(tensor):
    """反归一化用于可视化"""
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape((3, 1, 1))
    std = np.array([0.2023, 0.1994, 0.2010]).reshape((3, 1, 1))
    tensor = tensor.numpy() * std + mean
    tensor = np.clip(tensor, 0, 1)
    return np.transpose(tensor, (1, 2, 0))

def plot_metrics(metrics):
    """绘制量化评估图表"""
    labels = ['Clean Acc', 'Defended Clean Acc', 'Adv Acc', 'Defended Adv Acc']
    values = [metrics['clean_acc'], metrics['defended_clean_acc'], metrics['adv_acc'], metrics['defended_adv_acc']]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    bars = plt.bar(labels, values, color=['#4CAF50', '#81C784', '#F44336', '#E57373'])
    plt.xticks(rotation=15)
    plt.ylim(0, 1.1)
    plt.ylabel('Accuracy / Robustness')
    plt.title('Accuracy vs Robustness Trade-off')
    
    # 在柱状图上添加具体数值
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2%}', ha='center', va='bottom')

    # 绘制计算开销对比
    plt.subplot(1, 2, 2)
    latencies = [metrics['avg_clean_latency']*1000, metrics['avg_defend_latency']*1000]
    bars_lat = plt.bar(['Base Inference', 'Inference + IG Detection'], latencies, color=['#2196F3', '#FF9800'], width=0.5)
    plt.ylabel('Latency (ms)')
    plt.title('Inference Latency Overhead')
    
    max_latency = max(latencies)
    for bar in bars_lat:
        yval = bar.get_height()
        # 将原本的 yval + 0.5 改为 yval + max_latency * 0.05
        plt.text(bar.get_x() + bar.get_width()/2, yval + max_latency * 0.05, f'{yval:.2f} ms', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('evaluation_metrics.png', dpi=300)
    print("[*] 量化评估图表已保存至 evaluation_metrics.png")

def plot_images(results, classes, num_images=5):
    """绘制对比图像与四项核心预测结果"""
    plt.figure(figsize=(15, 7))
    
    for i in range(num_images):
        # 1. 干净样本展示区
        plt.subplot(2, num_images, i + 1)
        img_clean = unnormalize(results['clean_images'][i])
        plt.imshow(img_clean)
        
        true_label = classes[results['labels'][i]]
        clean_pred = classes[results['clean_preds'][i]]
        
        # 标题显示：真实标签和原始预测
        title_clean = f"True: {true_label}\nModel Pred: {clean_pred}"
        color_clean = 'green' if true_label == clean_pred else 'red'
        plt.title(title_clean, color=color_clean, fontsize=10)
        plt.axis('off')

        # 2. 对抗样本展示区
        plt.subplot(2, num_images, i + 1 + num_images)
        img_adv = unnormalize(results['adv_images'][i])
        plt.imshow(img_adv)
        
        adv_pred = classes[results['adv_preds'][i]]
        is_anomaly = results['adv_anomalies'][i].item()
        
        # 确定防御后的预测结果
        if is_anomaly:
            defended_pred = "Rejected (Anomaly)"
            color_adv = 'blue' # 防御成功，拦截
        else:
            defended_pred = adv_pred
            color_adv = 'red' # 防御失败，模型被欺骗
            
        # 标题显示：攻击后预测和防御后状态
        title_adv = f"Adv Pred: {adv_pred}\nDefended: {defended_pred}"
        plt.title(title_adv, color=color_adv, fontsize=10)
        plt.axis('off')
        
    plt.suptitle("Adversarial Attack and Defense Visualization", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('adversarial_examples.png', bbox_inches='tight', dpi=300)
    print("[*] 对抗样本及防御结果可视化已保存至 adversarial_examples.png")