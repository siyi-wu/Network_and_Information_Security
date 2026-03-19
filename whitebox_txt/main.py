import torch
from tqdm import tqdm
from dataset_prep import load_model_and_tokenizer, load_test_data
from attack_gen import gradient_based_word_swap
from evaluation import predict_text, calculate_word_change_rate, highlight_adversarial_text
from visualization import plot_nlp_tradeoff_curves, generate_html_report, plot_perceptual_tradeoff

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 使用设备: {device}")

    # 1. 准备模型和数据
    model, tokenizer = load_model_and_tokenizer(device=device)
    texts, labels = load_test_data(num_samples=100) # 取100条进行快速验证

    # 2. 攻击参数设置
    max_swaps_list = [1, 2, 3, 4, 5] 
    
    asr_results = []
    change_rate_results = []
    successful_cases_for_report = [] # 收集用于生成 HTML 的案例
    
    for max_swaps in max_swaps_list:
        print(f"\n{'-'*60}")
        print(f"[-] 正在测试攻击强度: 替换最多 {max_swaps} 个梯度显著词")
        
        total_correct_clean = 0
        successful_attacks = 0
        total_change_rate = 0.0
        
        for i in tqdm(range(len(texts)), desc=f"Attacking (swaps={max_swaps})"):
            text = texts[i]
            true_label = labels[i]
            
            # 测试干净样本准确性
            clean_pred = predict_text(model, tokenizer, text, device)
            if clean_pred != true_label:
                continue # 只攻击原本预测正确的样本
                
            total_correct_clean += 1
            
            # 生成对抗样本 (基于梯度的词替换)
            adv_text = gradient_based_word_swap(
                model, tokenizer, text, true_label, device, max_swaps=max_swaps
            )
            
            # 测试对抗样本
            adv_pred = predict_text(model, tokenizer, adv_text, device)
            
            if adv_pred != true_label:
                successful_attacks += 1
                total_change_rate += calculate_word_change_rate(text, adv_text)
                
                # 记录成功案例（每个 max_swaps 记录前2个即可，防止报告过长）
                if successful_attacks <= 2:
                    successful_cases_for_report.append({
                        'max_swaps': max_swaps,
                        'orig_text': text,
                        'adv_text': adv_text,
                        'true_label': true_label,
                        'adv_label': adv_pred
                    })
                
                # 在终端直观打印前几个成功的案例
                if successful_attacks == 1:
                    colored_adv_text = highlight_adversarial_text(text, adv_text)
                    print(f"\n\n[+] 成功案例展示 (Max Swaps: {max_swaps}):")
                    print(f"    原始文本 (标签 {clean_pred}): {text}")
                    print(f"    对抗文本 (标签 {adv_pred}): {colored_adv_text}\n")

        # 计算并记录指标
        if total_correct_clean > 0:
            asr = successful_attacks / total_correct_clean
            avg_change_rate = total_change_rate / successful_attacks if successful_attacks > 0 else 0
            
            asr_results.append(asr)
            change_rate_results.append(avg_change_rate)
            
            print(f"    -> [基线] 模型原始准确率: {total_correct_clean/len(texts):.2%}")
            print(f"    -> [攻击] 攻击成功率 (ASR): {asr:.2%}")
            print(f"    -> [相似性] 平均词汇修改率: {avg_change_rate:.2%}")

    # 3. 结果汇总与多维度可视化 
    print(f"\n{'-'*60}")
    print("[*] 正在生成实验量化评估图表...")
    
    # 原始 NLP 曲线
    plot_nlp_tradeoff_curves(max_swaps_list, asr_results, change_rate_results)
    
    # 新增：符合实验要求格式的 ASR vs 感知相似性权衡曲线 
    # 在文本任务中，change_rate_results 对应感知距离
    # 如果你在做图像实验，则传入 LPIPS/SSIM 的计算结果
    plot_perceptual_tradeoff(
        asr_list=asr_results, 
        similarity_metrics=change_rate_results, 
        metric_name='WordChangeRate_as_Perceptual' # 明确标注对应关系
    )
    
    generate_html_report(successful_cases_for_report)
    print("[*] 实验二量化评估与可视化任务全部完成！")

if __name__ == "__main__":
    main()