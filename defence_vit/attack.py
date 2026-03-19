# attack.py
import torch
import torch.nn as nn

def pgd_attack(model, images, labels, epsilon=8/255, alpha=2/255, iters=10, device='cuda'):
    """
    白盒 PGD (Projected Gradient Descent) 攻击 
    """
    images = images.clone().detach().to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()

    adv_images = images.clone().detach()
    adv_images.requires_grad = True

    for i in range(iters):
        outputs = model(adv_images)
        loss = loss_fn(outputs, labels)
        
        # 模型梯度清零
        model.zero_grad()
        loss.backward()

        # 生成扰动
        with torch.no_grad():
            adv_images = adv_images + alpha * adv_images.grad.sign()
            # 限制扰动范围在 [-epsilon, epsilon] 之间
            eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
            # 保证图像像素值依然在有效范围内
            adv_images = torch.clamp(images + eta, min=0, max=1)
        
        adv_images.requires_grad = True

    return adv_images