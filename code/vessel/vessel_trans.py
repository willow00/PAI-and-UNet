# 对10张血管图像进行数据增强，生成200张图像
import os
import random
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np

def vessel_transform(image, target):
    # 设置图像大小为256*256
    RS_trans = transforms.Resize([256, 256])
    image = RS_trans(image)
    target = RS_trans(target)

    # 随机透视变换
    if random.random() > 0.5:
        # [top - left, top - right, bottom - right, bottom - left]
        start = [[0, 0], [0, 255], [255, 0], [255, 255]]
        end = [[0, random.randint(0, 50)], [0, 255], [255, 0], [255, random.randint(200, 255)]]
        image = F.perspective(image, startpoints=start, endpoints=end, fill=255)
        target = F.perspective(target, startpoints=start, endpoints=end,fill=255)

    # 随机水平、垂直翻转
    RHF_trans = transforms.RandomHorizontalFlip(p=1)
    RVF_trans = transforms.RandomVerticalFlip(p=1)
    if random.random() > 0.5:
        image = RHF_trans(image)
        target = RHF_trans(target)

    if random.random() > 0.5:
        image = RVF_trans(image)
        target = RVF_trans(target)

    # 随机旋转
    angle = random.randint(-180, 180)
    image = F.rotate(image, angle, fill=255)
    target = F.rotate(target, angle, fill=255)

    return image, target

# 非理想检测条件下的重建图像
image_dir = "C:\\Users\\Jzq\\Desktop\\毕业论文\\pythonProject\\vessel_dataset\\train\\images"
# ground truth:
target_dir = "C:\\Users\\Jzq\\Desktop\\毕业论文\\pythonProject\\vessel_dataset\\train\\targets"

# 对重建图像及其ground truth进行相同的随机变换并保存：
for i in range(19):
    for j in range(10):
        img_path = os.path.join(image_dir, "{}.jpg".format(j+1))
        image = Image.open(img_path)
        target_path = os.path.join(target_dir, "{}.jpg".format(j+1))
        target = Image.open(target_path)

        image1, target1 = vessel_transform(image, target)

        save_path = os.path.join(image_dir, "{}.jpg".format((i+1)*10+j+1))
        image1.save(save_path)
        save_path = os.path.join(target_dir, "{}.jpg".format((i+1)*10+j+1))
        target1.save(save_path)
