import torchattacks
import config

def get_pgd_attacker(model):
    atk = torchattacks.PGD(model, eps=config.EPSILON, alpha=config.ALPHA, steps=config.STEPS)

    atk.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    return atk