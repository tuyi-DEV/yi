import os
import numpy as np
from PIL import Image
from utils.dataloader import DataGenerator
from utils.utils import cvtColor

# 测试验证集数据加载
def verify_val_data():
    # 读取验证集数据
    val_lines = []
    with open('cls_test.txt', 'r', encoding='utf-8') as f:
        val_lines = f.readlines()
    
    print(f"验证集总行数: {len(val_lines)}")
    
    # 创建数据生成器
    data_generator = DataGenerator(val_lines, [224, 224], random=False, autoaugment_flag=False, mosaic_flag=False)
    
    # 测试前10个样本
    for i in range(min(10, len(val_lines))):
        try:
            # 获取原始数据
            original_line = val_lines[i].strip()
            original_label = int(original_line.split(';')[0])
            original_path = original_line.split(';')[1].split()[0]
            
            # 从数据生成器获取数据
            image, label = data_generator[i]
            
            # 检查文件是否存在
            if os.path.exists(original_path):
                # 读取原始图像
                img = Image.open(original_path)
                img = cvtColor(img)
                
                print(f"样本 {i}:")
                print(f"  原始标签: {original_label}")
                print(f"  生成器标签: {label}")
                print(f"  路径: {original_path}")
                print(f"  文件存在: {os.path.exists(original_path)}")
                print(f"  图像形状: {img.size}")
                print(f"  生成器输出形状: {image.shape}")
                print(f"  标签是否匹配: {original_label == label}")
                print()
            else:
                print(f"样本 {i}: 文件不存在 - {original_path}")
                print()
                
        except Exception as e:
            print(f"样本 {i} 出错: {str(e)}")
            print()

if __name__ == '__main__':
    verify_val_data()
