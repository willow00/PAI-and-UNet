# 对shepp logan进行数据增强，生成200张图像
import os
import random
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

img_path = 'D:\\matlab_paper\\shepp_logan\\sl.jpg'
img_PIL = Image.open(img_path)

# 图片边缘用黑色填充，以便椭圆均位于传感器内
Pad_trans = transforms.Pad(padding=20)
img_PIL = Pad_trans(img_PIL)

# 将图片变为256*256大小：
Resize_trans = transforms.Resize(size=256)
img_PIL = Resize_trans(img_PIL)

# 保存大小为256的原始图片：
save_dir = "D:\\matlab_paper\\shepp_logan\\shepp_logan_pic"
save_path = os.path.join(save_dir, "1.jpg")
img_PIL.save(save_path)

# RandomHorizontalFlip:随机水平翻转
# RandomVerticalFlip:随机垂直翻转
# RandomPerspective：随机视角变化
# ColorJitter:随机更改图像的亮度、对比度和饱和度
# RandomRotation: 旋转

# 从transform中选择一个class，创建一个实例
RHF_trans = transforms.RandomHorizontalFlip()  # 以0.5的概率进行水平翻转
RVF_trans = transforms.RandomVerticalFlip()  # 以0.5的概率进行垂直翻转
# 图片随机旋转
angle = 180
RR_trans = transforms.RandomRotation(degrees=angle)
# 图片随机视角变化：以0.5的概率进行(0.2,0.6)中随机数的程度的扭曲变化
RP_trans = transforms.RandomPerspective(distortion_scale=0.4*random.random()+0.2)
# 图片亮度，对比度调整:
CJ_trans = transforms.ColorJitter(brightness=0.5, contrast=0.5)
# 组合变换：
trans_compose = transforms.Compose([RHF_trans, RVF_trans, RR_trans, RP_trans, CJ_trans])


# 对shepplogan原始图进行随机变换并保存：
for i in range(199):
    trans_img = trans_compose(img_PIL)
    save_path = os.path.join(save_dir, "{}.jpg".format(i+2))
    trans_img.save(save_path)
