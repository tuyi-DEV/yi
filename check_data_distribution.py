import os
import numpy as np
from PIL import Image

# 检查数据集分布
def check_data_distribution():
    # 检查训练集
    train_lines = []
    with open('cls_train.txt', 'r', encoding='utf-8') as f:
        train_lines = f.readlines()
    
    # 检查验证集
    val_lines = []
    with open('cls_test.txt', 'r', encoding='utf-8') as f:
        val_lines = f.readlines()
    
    print(f"训练集样本数: {len(train_lines)}")
    print(f"验证集样本数: {len(val_lines)}")
    
    # 统计训练集类别分布
    train_labels = []
    for line in train_lines:
        try:
            label = int(line.split(';')[0])
            train_labels.append(label)
        except:
            pass
    
    # 统计验证集类别分布
    val_labels = []
    for line in val_lines:
        try:
            label = int(line.split(';')[0])
            val_labels.append(label)
        except:
            pass
    
    print(f"训练集类别分布: 0={train_labels.count(0)}, 1={train_labels.count(1)}")
    print(f"验证集类别分布: 0={val_labels.count(0)}, 1={val_labels.count(1)}")
    
    # 检查图像大小分布
    print("\n检查图像大小分布:")
    
    # 训练集图像大小
    train_sizes = []
    for line in train_lines[:100]:  # 只检查前100张
        try:
            path = line.split(';')[1].split()[0]
            if os.path.exists(path):
                img = Image.open(path)
                train_sizes.append(img.size)
        except:
            pass
    
    # 验证集图像大小
    val_sizes = []
    for line in val_lines[:100]:  # 只检查前100张
        try:
            path = line.split(';')[1].split()[0]
            if os.path.exists(path):
                img = Image.open(path)
                val_sizes.append(img.size)
        except:
            pass
    
    if train_sizes:
        train_size_counts = {}
        for size in train_sizes:
            if size in train_size_counts:
                train_size_counts[size] += 1
            else:
                train_size_counts[size] = 1
        print(f"训练集前100张图像大小分布: {train_size_counts}")
    
    if val_sizes:
        val_size_counts = {}
        for size in val_sizes:
            if size in val_size_counts:
                val_size_counts[size] += 1
            else:
                val_size_counts[size] = 1
        print(f"验证集前100张图像大小分布: {val_size_counts}")

if __name__ == '__main__':
    check_data_distribution()
