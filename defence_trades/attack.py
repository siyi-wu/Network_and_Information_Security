import torch
import torch.nn.functional as F

def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=20):
    # 如果 images 已经归一化，PGD 产生的扰动也必须在归一化空间内
    images = images.clone().detach()
    adv_images = images.clone().detach() + torch.empty_like(images).uniform_(-eps, eps)
    
    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = F.cross_entropy(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images)[0]
        adv_images = adv_images.detach() + alpha * grad.sign()
        # 限制在 epsilon 范围内
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = images + delta
    return adv_images