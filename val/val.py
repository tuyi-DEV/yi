import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nets import get_model_from_name
from utils.dataloader import DataGenerator, detection_collate
from utils.utils import get_classes


def parse_args():
    parser = argparse.ArgumentParser(description='Image Classification Validation')
    parser.add_argument('--model-path', type=str, default='logs/best_model.pth', help='Path to model weights')
    parser.add_argument('--classes-path', type=str, default='model_data/cls_classes.txt', help='Path to classes file')
    parser.add_argument('--model', type=str, default='mobilenetv2', help='Model backbone')
    parser.add_argument('--input-shape', type=int, nargs=2, default=[224, 224], help='Input image shape')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for validation')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA if available')
    parser.add_argument('--test-file', type=str, default='cls_test.txt', help='Path to test annotation file')
    return parser.parse_args()


def validate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0
    loss_function = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            loss = loss_function(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == targets).sum().item()
            total_samples += targets.size(0)
    
    accuracy = total_correct / total_samples
    avg_loss = total_loss / len(dataloader)
    return accuracy, avg_loss


def main():
    args = parse_args()
    
    # 获取设备
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 获取类别
    classes, num_classes = get_classes(args.classes_path)
    print(f'Classes: {classes}')
    print(f'Number of classes: {num_classes}')
    
    # 创建模型
    model = get_model_from_name(args.model)(num_classes=num_classes, pretrained=False)
    
    # 加载权重
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f'Model weights loaded from {args.model_path}')
    else:
        print(f'Model weights not found at {args.model_path}')
        return
    
    # 将模型移到设备
    model = model.to(device)
    
    # 加载验证数据
    with open(args.test_file, encoding='utf-8') as f:
        test_lines = f.readlines()
    
    if len(test_lines) == 0:
        print('Test file is empty')
        return
    
    print(f'Number of validation samples: {len(test_lines)}')
    
    # 创建数据生成器
    val_dataset = DataGenerator(test_lines, args.input_shape, args.batch_size, False)
    val_dataloader = DataLoader(
        val_dataset, 
        shuffle=False, 
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        pin_memory=True,
        drop_last=False, 
        collate_fn=detection_collate
    )
    
    # 进行验证
    print('Starting validation...')
    accuracy, avg_loss = validate(model, val_dataloader, device)
    
    # 输出验证结果
    print('Validation Results:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Average Loss: {avg_loss:.4f}')
    print(f'Correct: {int(accuracy * len(test_lines))}/{len(test_lines)}')


if __name__ == '__main__':
    main()
