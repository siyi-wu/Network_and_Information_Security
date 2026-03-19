import torch
import torch.nn.functional as F

def pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=20):
    # 这里的 images 应该是已经过 Normalize 的，或者在攻击中处理
    # 建议：在 evaluate 函数中传入 images 之前不要做额外的处理
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
        # 注意：如果做了 Normalize，这里的 clamp 就不能简单用 [0,1]
    return adv_images