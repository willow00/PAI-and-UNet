# 训练好的unet用于重建
import scipy.io as io
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
# 用于计时的包
import time
from Unetmodel import *
from loadmydataset import *

# 超参数的确定：
batch_size0 = 1

# 要重建的数据集所在文件夹：
test_img_dir = "..\\dataset\\test\\images"
test_target_dir = "..\\dataset\\test\\targets"

# 确定训练设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("训练设备:{}".format(device))

# 加载数据集
test_data = MyDataset(test_img_dir, test_target_dir, train_flag=False, channel=3)

# 看数据集的大小
test_data_size = len(test_data)
print("数据集的长度为：{}".format(test_data_size))

# 用dataloader来加载数据
test_dataloader = DataLoader(test_data, batch_size=batch_size0, shuffle=False)


# 加载网络模型
unet = Unet2()
unet.load_state_dict(torch.load("unet_70.pth"))
# 将模型转移至设备上
unet.to(device)

# 定义损失函数MSE：(可自己定义其他的损失函数)
loss_fn = nn.MSELoss()
# 将损失函数转移至设备上
loss_fn.to(device)


# 设置验证模式:
unet.eval()
total_test_loss1 = 0
total_test_loss2 = 0
with torch.no_grad():
    n = 0  # 协助输出结果保存命名
    for data in test_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)

        outputs = unet(imgs)

        # 保存输出的tensor数据为matlab矩阵，然后在matlab中将其转为图片（见文件mat2pic.m）
        for j in range(batch_size0):
            # 保存路径：
            save_dir = "..\\discdata\\new_unet1\\train"
            # 注意将数据转到cpu上
            result = np.array(outputs[j].cpu())
            io.savemat(os.path.join(save_dir, "{}.mat".format(j + 1 + n * batch_size0)), {'result': result})
        n = n + 1

        # 经unet后处理数据集的损失函数
        loss = loss_fn(outputs, targets)
        total_test_loss1 += loss

        # unet后处理前数据集的损失函数
        loss = loss_fn(imgs, targets)
        total_test_loss2 += loss

    # 打印结果:
    print("经unet后处理数据集上的loss：{}".format(total_test_loss1))
    print("原始数据集上的loss：{}".format(total_test_loss2))

