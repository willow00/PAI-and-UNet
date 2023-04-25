# 用shepplogan图像对灰度圆盘数据集上训练好的unet模型进行微调（fine-tuning）:
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
# 用于计时的包
import time
# 将网络模型加载进来
from Unetmodel import *
from loadmydataset import *

# 超参数的确定：
learning_rate = 0.00001  # 比预训练的小10倍
batch_size0 = 2
epoch = 40  # 调整1000次
weight_decay0 = 0.001

# 数据集所在文件夹：
train_img_dir = "../shepplogandataset/finetuning/images"
train_target_dir = "../shepplogandataset/finetuning/targets"

# 确定训练设备
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("训练设备:{}".format(device))

# # 步骤1：准备数据集
# 注：图片大小默认为256*256，如需改变，传入参数
# 训练数据集：
train_data = MyDataset(train_img_dir, train_target_dir, train_flag=True, channel=1)

# 看数据集的大小
train_data_size = len(train_data)
print("训练数据集的长度为：{}".format(train_data_size))

# 步骤2：用dataloader来加载数据（这里注意一下会不会舍掉除不尽的部分吧，以及是否打乱等其他参数）
# shuffer=Ture表示在每一次epoch中都打乱所有数据的顺序
train_dataloader = DataLoader(train_data, batch_size=batch_size0, shuffle=True)

# 步骤3：搭建神经网络，已在开头import

# 步骤4：创建网络模型
# 灰度图像通道为1
unet = Unet2(original_channels=1)
# 加载模型参数：
unet.load_state_dict(torch.load("unet_shepplogan.pth"))
# 将模型转移至设备上
unet.to(device)

# 步骤5：定义损失函数MSE：(可自己定义其他的损失函数)
loss_fn = nn.MSELoss()
# 将损失函数转移至设备上
loss_fn.to(device)

# 步骤6：设置Adam优化器
# beta1 = 0.9
# beta2 = 0.999
optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate, weight_decay=weight_decay0)

# 步骤7：训练网络
# 设置训练的参数:
# 记录训练的次数
total_train_step = 0

# 记录开始时间
start_time = time.time()

for i in range(epoch):

    print("----------第 {} 轮训练开始----------".format(i + 1))

    # 训练开始
    # 设置训练模式:
    unet.train()

    for data in train_dataloader:
        imgs, targets = data
        # 将数据转移至设备上
        imgs = imgs.to(device)
        targets = targets.to(device)

        # 输入神经网络得到输出：
        outputs = unet(imgs)

        # 计算损失函数
        loss = loss_fn(outputs, targets)

        # 优化器优化模型：梯度下降
        # 梯度置零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()

        total_train_step += 1
        # 每隔50步打印：
        if total_train_step % 50 == 0:
            # 记录训练50步的结束时间
            end_time = time.time()
            # 打印训练50步所用时间
            print("训练所用时间:{}".format(end_time - start_time))
            # 打印训练50步的损失函数值
            print("训练次数：{}，loss：{}".format(total_train_step, loss.item()))


    # 保存训练最终得到的模型
    if i == epoch-1:
        torch.save(unet.state_dict(), "unet_shepplogan_ft{}.pth".format(i))
        print("模型已保存")
