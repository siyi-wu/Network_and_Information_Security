import os

# 路径配置
MODEL_DIR = './torch'
OUTPUT_DIR = './outputs'
DATA_DIR = './data'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# 攻击与训练参数
PATCH_SIZES = [2, 4, 6, 8, 10]  # 用于绘制权衡曲线的多个补丁尺寸
TARGET_CLASS = 0             
BATCH_SIZE = 128             
EPOCHS = 5                   # 为了快速跑完多个尺寸，将Epoch稍微调小
LEARNING_RATE = 0.05         

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')