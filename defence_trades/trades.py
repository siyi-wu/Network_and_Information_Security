import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def trades_loss(model, x_natural, y, optimizer, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=1.0):
    """TRADES 损失函数与 KL 对抗样本生成"""
    model.eval()
    batch_size = len(x_natural)
    
    # 随机初始化扰动
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(x_natural.device).detach()
    
    # 基于 KL 散度生成对抗样本 (针对 TRADES 优化)
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                               F.softmax(model(x_natural), dim=1),
                               reduction='sum')
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    model.train()
    x_adv = torch.autograd.Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    
    # 计算最终损失
    logits = model(x_natural)
    loss_natural = F.cross_entropy(logits, y)
    loss_robust = (1.0 / batch_size) * F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                                F.softmax(model(x_natural), dim=1),
                                                reduction='sum')
    loss = loss_natural + beta * loss_robust
    return loss