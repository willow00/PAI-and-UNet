# 加载自己的数据集
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np

class MyDataset(Dataset):
    def __init__(self, images_dir, targets_dir, train_flag=True, image_height=256, image_width=256, channel=3):
        # 图像所在文件夹：
        self.images_dir = images_dir
        # 初始压力分布所在文件夹：
        self.targets_dir = targets_dir
        # 图像名称组成的序列：
        self.images = os.listdir(images_dir)
        # 按文件名排序：
        self.images.sort(key=lambda x: int(x.split('.')[0]))
        # 用于判断是否为训练数据：
        self.train_flag = train_flag
        # 图像通道数：
        self.channel = channel


        # 这里只改变训练集和验证集的大小，并转为tensor
        # 对训练数据的处理：
        self.train_tf = transforms.Compose([
            transforms.Resize([image_height, image_width]),
            transforms.ToTensor()
        ])

        # 对测试数据的处理：
        self.val_tf = transforms.Compose([
            transforms.Resize([image_height, image_width]),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # 图片所在路径：
        img_path = os.path.join(self.images_dir, self.images[index])
        # 初始压力所在路径：
        target_path = os.path.join(self.targets_dir, self.images[index])
        
        # 加载RGB图像：
        if self.channel == 3:
            image = Image.open(img_path).convert("RGB")
            target = Image.open(target_path).convert("RGB")

        # 加载灰度图像：
        if self.channel == 1:
            image = Image.open(img_path).convert("L")
            target = Image.open(target_path).convert("L")

        # 对其实行相应的transform，主要是将其变为tensor
        if self.train_flag:
            image = self.train_tf(image)
            target = self.train_tf(target)
        else:
            image = self.val_tf(image)
            target = self.val_tf(target)
        return image, target

