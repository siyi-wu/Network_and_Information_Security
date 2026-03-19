# visualize.py
import matplotlib.pyplot as plt
import numpy as np

def visualize_defense_and_tradeoff(clean_img, adv_img, purified_img, true_label, 
                                   base_pred, adv_pred, def_pred,
                                   base_clean_acc, base_adv_acc, def_clean_acc, def_adv_acc):
    """
    可视化防御成功的案例，并同时绘制精度权衡柱状图 (纯英文版，适配 Linux 无中文字体环境)
    """
    clean_img_np = clean_img.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    adv_img_np = adv_img.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    purified_img_np = purified_img.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    
    # 放大噪声以便肉眼观察
    noise = np.clip((adv_img_np - clean_img_np) * 10 + 0.5, 0, 1)

    fig = plt.figure(figsize=(14, 10))
    
    # ================= 区域 1：图像可视化 (上半部分) =================
    ax1 = plt.subplot(2, 4, 1)
    ax1.imshow(clean_img_np)
    ax1.set_title(f"Clean Image\nTrue Label: {true_label}\nBase Pred: {base_pred}", color='green')
    ax1.axis('off')
    
    ax2 = plt.subplot(2, 4, 2)
    ax2.imshow(noise)
    ax2.set_title("PGD Perturbation\n(Amplified)")
    ax2.axis('off')
    
    ax3 = plt.subplot(2, 4, 3)
    ax3.imshow(adv_img_np)
    ax3.set_title(f"Adversarial Sample\nBase Pred: {adv_pred}", color='red')
    ax3.axis('off')
    
    ax4 = plt.subplot(2, 4, 4)
    ax4.imshow(purified_img_np)
    ax4.set_title(f"Purified by Defense\nDefended Pred: {def_pred}", color='green')
    ax4.axis('off')

    # ================= 区域 2：防御代价与鲁棒性权衡图表 (下半部分) =================
    ax5 = plt.subplot(2, 1, 2)
    
    # 移除中文，避免字体缺失警告
    labels = ['Clean Sample Accuracy', 'Robust Accuracy']
    base_scores = [base_clean_acc, base_adv_acc]
    def_scores = [def_clean_acc, def_adv_acc]
    
    x = np.arange(len(labels))
    width = 0.35
    
    # 移除中文图例
    rects1 = ax5.bar(x - width/2, base_scores, width, label='Base ViT', color='#ff9999')
    rects2 = ax5.bar(x + width/2, def_scores, width, label='Defended ViT', color='#66b3ff')
    
    ax5.set_ylabel('Accuracy (%)')
    ax5.set_title('Defense Trade-off: Clean Acc Drop vs Robustness Gain')
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels)
    ax5.set_ylim(0, 110) # 留出顶部空间显示数值
    ax5.legend()
    
    # 在柱子上添加具体的数值标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax5.annotate(f'{height:.2f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
            
    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig("defense_success_and_tradeoff.png", dpi=300)
    print("\n[✔] 成功生成综合报告图！已保存为: defense_success_and_tradeoff.png")