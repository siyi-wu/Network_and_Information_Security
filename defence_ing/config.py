import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
HUB_DIR = './torch' # 模型保存路径
DATA_DIR = './data'

# 攻击参数
EPSILON = 8 / 255
ALPHA = 2 / 255
STEPS = 10

# 防御参数：IG 异常检测的阈值 (需根据实际分布微调)
IG_THRESHOLD = 0.05