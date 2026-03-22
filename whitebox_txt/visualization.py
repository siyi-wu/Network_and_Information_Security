import matplotlib.pyplot as plt
import os

def plot_perceptual_tradeoff(asr_list, similarity_metrics, metric_name='LPIPS', save_dir='./results'):
    """
    绘制 ASR 与感知相似性(LPIPS/SSIM/Change Rate)的权衡曲线 
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    # 绘制曲线
    plt.plot(similarity_metrics, asr_list, marker='D', color='forestgreen', 
             linestyle='--', linewidth=2, markersize=8)
    
    # 添加标注（针对量化评估要求） 
    plt.title(f'Trade-off Curve: ASR vs {metric_name}')
    plt.xlabel(f'Perceptual Distance / Similarity ({metric_name})')
    plt.ylabel('Attack Success Rate (ASR)')
    
    # 增加网格和背景美化
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3) # 50% ASR 参考线
    
    save_path = os.path.join(save_dir, f'perceptual_tradeoff_{metric_name}.png')
    plt.savefig(save_path, dpi=300)
    print(f"[*] 量化评估权衡曲线 ({metric_name}) 已保存至: {save_path}")
    plt.close()

def plot_nlp_tradeoff_curves(max_swaps_list, asr_list, change_rate_list, save_dir='./results'):
    """
    绘制 NLP 对抗攻击的量化评估曲线并保存。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))

    # 子图 1: 扰动强度 (Max Swaps) vs ASR
    plt.subplot(1, 2, 1)
    plt.plot(max_swaps_list, asr_list, marker='o', color='red', linestyle='-', linewidth=2)
    plt.title('Attack Strength vs ASR')
    plt.xlabel('Max Words Swapped')
    plt.ylabel('Attack Success Rate (ASR)')
    plt.xticks(max_swaps_list)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 子图 2: 词汇修改率 vs ASR (权衡曲线)
    plt.subplot(1, 2, 2)
    plt.plot(change_rate_list, asr_list, marker='s', color='blue', linestyle='-', linewidth=2)
    for i, txt in enumerate(max_swaps_list):
        plt.annotate(f'swaps={txt}', (change_rate_list[i], asr_list[i]), 
                     textcoords="offset points", xytext=(0,10), ha='center')
    plt.title('Trade-off: Word Change Rate vs ASR')
    plt.xlabel('Average Word Change Rate')
    plt.ylabel('Attack Success Rate (ASR)')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'nlp_tradeoff_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"[*] NLP 评估曲线已保存至: {save_path}")
    plt.close()

def generate_html_report(successful_cases, save_dir='./results'):
    """
    生成直观的 HTML 报告，用于展示攻击前后的语句变化。
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'adversarial_examples_report.html')
    
    html_content = """
    <html>
    <head>
        <title>Adversarial Attack Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #dddddd; text-align: left; padding: 8px; }
            th { background-color: #f2f2f2; }
            .unk { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <h2>NLP Adversarial Examples (Gradient-Based Word Swap)</h2>
        <table>
            <tr>
                <th>Max Swaps</th>
                <th>Original Text (Label)</th>
                <th>Adversarial Text (Prediction)</th>
            </tr>
    """
    
    for case in successful_cases:
        adv_html = case['adv_text'].replace("[UNK]", "<span class='unk'>[UNK]</span>")
        html_content += f"""
            <tr>
                <td>{case['max_swaps']}</td>
                <td>{case['orig_text']}<br><i>True Label: {case['true_label']}</i></td>
                <td>{adv_html}<br><i>Adv Label: {case['adv_label']}</i></td>
            </tr>
        """
        
    html_content += """
        </table>
    </body>
    </html>
    """
    
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"[*] 对抗样本 HTML 可视化报告已保存至: {save_path}")