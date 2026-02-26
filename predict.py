import os
import argparse
from PIL import Image

from classification import Classification


def parse_args():
    parser = argparse.ArgumentParser(description='Single Image Classification')
    #------------------------------------------------------#
    #   image_path: 这里是用户输入图片路径的参数名称
    #   使用方法: python predict.py path/to/your/image.jpg
    #   例如: python predict.py datasets/train/Positive/15.jpg
    #------------------------------------------------------#
    parser.add_argument('image_path', type=str, help='Path to the image file (required)')
    parser.add_argument('--model-path', type=str, default='logs/best_epoch_weights.pth', help='Path to model weights')
    parser.add_argument('--classes-path', type=str, default='model_data/cls_classes.txt', help='Path to classes file')
    parser.add_argument('--backbone', type=str, default='mobilenetv2', help='Model backbone')
    parser.add_argument('--input-shape', type=int, nargs=2, default=[224, 224], help='Input image shape')
    parser.add_argument('--letterbox-image', action='store_true', default=False, help='Use letterbox image resize')
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA if available')
    parser.add_argument('--show-image', action='store_true', default=False, help='Show the image with prediction result')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查图片文件是否存在
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found at {args.image_path}")
        return
    
    # 初始化分类器
    print("Initializing classifier...")
    classifier = Classification(
        model_path=args.model_path,
        classes_path=args.classes_path,
        backbone=args.backbone,
        input_shape=args.input_shape,
        letterbox_image=args.letterbox_image,
        cuda=args.cuda
    )
    
    # 加载图片
    print(f"Loading image: {args.image_path}")
    try:
        image = Image.open(args.image_path)
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        return
    
    # 进行预测
    print("Predicting...")
    class_name = classifier.detect_image(image)
    
    # 输出结果
    print(f"Prediction result: {class_name}")


if __name__ == '__main__':
    main()
