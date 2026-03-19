import torchattacks

def get_pgd_attacker(model, eps=8/255, alpha=2/255, steps=10):
    """
    初始化 PGD 攻击器。
    """
    # 初始化 PGD 攻击器，使用 L_inf 范数
    atk = torchattacks.PGD(model, eps=eps, alpha=alpha, steps=steps)
    atk.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    return atk

def generate_adv_images(atk, images, labels):
    """生成对抗样本"""
    adv_images = atk(images, labels)
    return adv_images