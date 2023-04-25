# U_net模型
import torch
import torch.nn as nn
from torchsummary import summary

# 进行两次“卷积-激活函数ReLU”：两个深蓝色箭头
class DoubleConv(nn.Module):
    """(convolution => BatchNorm => ReLU) * 2"""

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, stride=1),
            # 注：进行填充，方便上采样复制连接，与原文不同！
            # 加入正则化层,因此多了一个参数mid_channels, mid_channels=out_channels
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# 定义U-net网络框架
class Unet2(nn.Module):

    def __init__(self, original_channels=3):
        super(Unet2, self).__init__()
        self.original_channels = original_channels  # 输入图像和输出图像的通道，RGB为3，灰度图像为1

        # 左边的部分：
        # 两个蓝色箭头：
        self.encoder1 = DoubleConv(self.original_channels, mid_channels=64, out_channels=64)
        # 一个红色箭头：
        self.down1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = DoubleConv(in_channels=64, mid_channels=128, out_channels=128)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = DoubleConv(in_channels=128, mid_channels=256, out_channels=256)
        self.down3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder4 = DoubleConv(in_channels=256, mid_channels=512, out_channels=512)
        self.down4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder5 = DoubleConv(in_channels=512, mid_channels=1024, out_channels=1024)

        # 右边的部分：
        # 绿色箭头：
        self.up1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        # 两个蓝色箭头：
        self.decoder1 = DoubleConv(in_channels=1024, mid_channels=512, out_channels=512)
        # 注：灰色箭头的裁剪复制放在forward中实现

        self.up2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(in_channels=512, mid_channels=256, out_channels=256)

        self.up3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(in_channels=256, mid_channels=128, out_channels=128)

        self.up4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(in_channels=128, mid_channels=64, out_channels=64)

        self.decoder5 = nn.Conv2d(in_channels=64, out_channels=self.original_channels, kernel_size=1)

    # 定义正向传播过程：
    # 注：要求输入x是四维度的(N,C,H,W)
    def forward(self, x):
        # 左边部分：
        encoder1 = self.encoder1(x)
        down1 = self.down1(encoder1)

        encoder2 = self.encoder2(down1)
        down2 = self.down2(encoder2)

        encoder3 = self.encoder3(down2)
        down3 = self.down3(encoder3)

        encoder4 = self.encoder4(down3)
        down4 = self.down4(encoder4)

        encoder5 = self.encoder5(down4)

        # 右边部分：
        up1 = self.up1(encoder5)
        # 用torch.cat进行复制连接
        # 四维数据：(batch size, channels, height, width), 其中第二维度是通道数(参数dim从0开始算)
        decoder1 = self.decoder1(torch.cat((encoder4, up1), dim=1))

        up2 = self.up2(decoder1)
        decoder2 = self.decoder2(torch.cat((encoder3, up2), dim=1))

        up3 = self.up3(decoder2)
        decoder3 = self.decoder3(torch.cat((encoder2, up3), dim=1))

        up4 = self.up4(decoder3)
        decoder4 = self.decoder4(torch.cat((encoder1, up4), dim=1))

        # 输出
        out = self.decoder5(decoder4)
        return out

# 神经网络每一层的输出大小和参数数量
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
unet2 = Unet2(original_channels=3).to(device)
summary(unet2, input_size=(3, 256, 256))

# # 神经网络的正确性检验：给定一个输入，看输出的尺寸是否正确
# if __name__ == '__main__':
#     unet2 = Unet2(original_channels=3)
#     input = torch.ones(64, 3, 32, 32)
#     output = unet2(input)
#     print(output.shape)