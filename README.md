# PAI-and-UNet
基于深度学习中U-Net在光声成像中的探讨
1.	code文件夹提供了文章中使用的数据集，时间反演算法重建代码以及利用U-Net对重建图像进行后处理的代码。光声成像的模拟和时间反演算法重建在MATLAB R2022a中实现，U-Net后处理利用Pytorch在   python3.7中实现。
2.	disc, vessel, shepplogan文件夹中分别保存了圆盘数据集、血管数据集、Shepp-Logan数据集上的时间反演算法(TR)代码实现，以及划分好训练集、验证集、测试集（和微调集）的数据集压缩包。
3.	Assessment文件夹中提供了利用matlab计算四个指标MSE，SSIM，PSNR，PCC的值的代码文件。
4.	post_processing_result 文件夹保存了三个数据集重建图像经U-Net后处理的结果。
5.	trained_unet.zip保存了在各数据集上训练好的U-Net模型的参数。
