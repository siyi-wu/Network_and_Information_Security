from config import *
from dataset import get_dataloader
from model_loader import load_resnet20
from attacker import get_pgd_attacker
from defender import IGDetector
from evaluator import evaluate_pipeline
from visualizer import plot_metrics, plot_images

def main():
    print("1. 加载数据与模型...")
    dataloader, classes = get_dataloader()
    model = load_resnet20()
    
    print("2. 初始化攻击器与防御器...")
    attacker = get_pgd_attacker(model)
    detector = IGDetector(model)
    
    print("3. 开始白盒模拟攻击与防御评估 (这可能需要几分钟)...")
    metrics, results = evaluate_pipeline(model, dataloader, attacker, detector)
    
    print("\n--- 量化评估结果 ---")
    print(f"原始干净样本精度: {metrics['clean_acc']:.2%}")
    print(f"防御后干净样本精度 (精度下降): {metrics['defended_clean_acc']:.2%} (下降了 {metrics['clean_acc'] - metrics['defended_clean_acc']:.2%})")
    print(f"对抗攻击成功率 (原始鲁棒性): {metrics['adv_acc']:.2%}")
    print(f"拦截后鲁棒性 (鲁棒性提升): {metrics['defended_adv_acc']:.2%} (提升了 {metrics['defended_adv_acc'] - metrics['adv_acc']:.2%})")
    print(f"单张样本基础推理延迟: {metrics['avg_clean_latency']*1000:.2f} ms")
    print(f"单张样本引入防御后延迟: {metrics['avg_defend_latency']*1000:.2f} ms (开销倍数: {metrics['avg_defend_latency']/metrics['avg_clean_latency']:.2f}x)")
    
    print("\n4. 生成可视化图表...")
    plot_metrics(metrics)
    plot_images(results, classes)

if __name__ == "__main__":
    main()