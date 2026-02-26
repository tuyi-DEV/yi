import os

# 数据集根目录
dataset_dir = 'datasets/test'
# 输出文件
output_file = 'cls_test.txt'

# 类别映射（注意文件夹名称的拼写）
class_map = {
    'test_negative': 0,  # 0 表示 negative
    'test_postive': 1    # 1 表示 positive
}

# 收集所有图片
all_images = []
for class_name, class_id in class_map.items():
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.exists(class_dir):
        images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]
        for img_path in images:
            # 确保路径使用正确的相对路径格式
            relative_path = os.path.relpath(img_path, os.path.abspath('.'))
            # 替换反斜杠为斜杠，确保路径格式一致
            relative_path = relative_path.replace('\\', '/')
            all_images.append(f"{class_id}; {relative_path}")

# 写入文件
with open(output_file, 'w', encoding='utf-8') as f:
    for line in all_images:
        f.write(line + '\n')

print(f"已生成测试集文件 {output_file}，包含 {len(all_images)} 张图片")
print(f"- Negative 类: {len([img for img in all_images if img.startswith('0;')])} 张")
print(f"- Positive 类: {len([img for img in all_images if img.startswith('1;')])} 张")
