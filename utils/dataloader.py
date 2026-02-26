import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from .utils import cvtColor, preprocess_input
from .utils_aug import CenterCrop, ImageNetPolicy, RandomResizedCrop, Resize


class DataGenerator(data.Dataset):
    def __init__(self, annotation_lines, input_shape, random=True, autoaugment_flag=True, mosaic_flag=False):
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.random             = random
        
        self.autoaugment_flag   = autoaugment_flag
        self.mosaic_flag        = mosaic_flag
        if self.autoaugment_flag:
            self.resize_crop = RandomResizedCrop(input_shape)
            self.policy      = ImageNetPolicy()
            
            self.resize      = Resize(input_shape[0] if input_shape[0] == input_shape[1] else input_shape)
            self.center_crop = CenterCrop(input_shape)

    def __len__(self):
        return len(self.annotation_lines)

    def __getitem__(self, index):
        # 尝试获取有效的图像路径
        while True:
            annotation_path = self.annotation_lines[index].split(';')[1].split()[0]
            # 检查文件是否存在
            if os.path.exists(annotation_path):
                break
            # 如果文件不存在，随机选择另一个索引
            index = np.random.randint(0, len(self.annotation_lines))
        
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image = Image.open(annotation_path)
        image = cvtColor(image)
        
        #------------------------------#
        #   Mosaic增强
        #------------------------------#
        if self.mosaic_flag and self.random:
            image, y = self.MosaicAugment(index)
        else:
            if self.autoaugment_flag:
                image = self.AutoAugment(image, random=self.random)
            else:
                image = self.get_random_data(image, self.input_shape, random=self.random)
            image = np.transpose(preprocess_input(np.array(image).astype(np.float32)), [2, 0, 1])
            y = int(self.annotation_lines[index].split(';')[0])
        
        return image, y

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            return image_data

        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.75, 1.5)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        rotate = self.rand()<.5
        if rotate: 
            angle = np.random.randint(-15,15)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue=[128, 128, 128]) 

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)
        return image_data
    
    def AutoAugment(self, image, random=True):
        if not random:
            image = self.resize(image)
            image = self.center_crop(image)
            return image

        #------------------------------------------#
        #   resize并且随即裁剪
        #------------------------------------------#
        image = self.resize_crop(image)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   随机增强
        #------------------------------------------#
        image = self.policy(image)
        return image
    
    def MosaicAugment(self, index):
        #------------------------------#
        #   选择4张图像
        #------------------------------#
        indices = [index]
        for _ in range(3):
            while True:
                idx = np.random.randint(0, len(self.annotation_lines))
                if idx not in indices:
                    # 检查文件是否存在
                    annotation_path = self.annotation_lines[idx].split(';')[1].split()[0]
                    if os.path.exists(annotation_path):
                        indices.append(idx)
                        break
        
        #------------------------------#
        #   读取4张图像
        #------------------------------#
        images = []
        labels = []
        for idx in indices:
            annotation_path = self.annotation_lines[idx].split(';')[1].split()[0]
            image = Image.open(annotation_path)
            image = cvtColor(image)
            images.append(image)
            labels.append(int(self.annotation_lines[idx].split(';')[0]))
        
        #------------------------------#
        #   创建马赛克图像
        #------------------------------#
        h, w = self.input_shape
        mosaic_image = np.zeros((h * 2, w * 2, 3), dtype=np.uint8) + 128
        
        #------------------------------#
        #   放置4张图像
        #------------------------------#
        positions = [(0, 0), (0, w), (h, 0), (h, w)]
        for i, (image, label) in enumerate(zip(images, labels)):
            # 随机缩放
            scale = self.rand(0.5, 1.5)
            new_h = int(image.height * scale)
            new_w = int(image.width * scale)
            image = image.resize((new_w, new_h), Image.BICUBIC)
            
            # 随机翻转
            if self.rand() < 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 随机旋转
            if self.rand() < 0.5:
                angle = np.random.randint(-15, 15)
                image = image.rotate(angle, expand=False)
            
            # 转换为numpy数组
            image = np.array(image)
            
            # 计算放置位置
            pos_h, pos_w = positions[i]
            img_h, img_w = image.shape[:2]
            
            # 计算目标区域
            start_h = max(0, pos_h - img_h // 2)
            start_w = max(0, pos_w - img_w // 2)
            end_h = min(h * 2, start_h + img_h)
            end_w = min(w * 2, start_w + img_w)
            
            # 计算源区域
            src_start_h = 0
            src_start_w = 0
            src_end_h = end_h - start_h
            src_end_w = end_w - start_w
            
            # 确保源区域不超出图像范围
            src_end_h = min(src_end_h, img_h)
            src_end_w = min(src_end_w, img_w)
            end_h = start_h + src_end_h
            end_w = start_w + src_end_w
            
            # 放置图像
            mosaic_image[start_h:end_h, start_w:end_w] = image[src_start_h:src_end_h, src_start_w:src_end_w]
        
        #------------------------------#
        #   随机裁剪
        #------------------------------#
        crop_h = np.random.randint(0, h)
        crop_w = np.random.randint(0, w)
        mosaic_image = mosaic_image[crop_h:crop_h + h, crop_w:crop_w + w]
        
        #------------------------------#
        #   随机颜色变换
        #------------------------------#
        if self.rand() < 0.5:
            # 色域变换
            r = np.random.uniform(-1, 1, 3) * [0.1, 1.5, 1.5] + 1
            hue, sat, val = cv2.split(cv2.cvtColor(mosaic_image, cv2.COLOR_RGB2HSV))
            dtype = mosaic_image.dtype
            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
            mosaic_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
            mosaic_image = cv2.cvtColor(mosaic_image, cv2.COLOR_HSV2RGB)
        
        #------------------------------#
        #   预处理
        #------------------------------#
        mosaic_image = np.transpose(preprocess_input(np.array(mosaic_image).astype(np.float32)), [2, 0, 1])
        
        #------------------------------#
        #   选择标签（使用第一个图像的标签）
        #------------------------------#
        return mosaic_image, labels[0]
            
def detection_collate(batch):
    images = []
    targets = []
    for image, y in batch:
        images.append(image)
        targets.append(y)
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    targets = torch.from_numpy(np.array(targets)).type(torch.FloatTensor).long()
    return images, targets
