%% ssim
% 数据类型为uint8，范围为0-255的图像image1,image2
% 或者 数据类型为float64，范围为0-1.0的图像image1,image2
clear; clc;
Total_num = 1000;
ssim_array = zeros(Total_num,1);
for j = 1 : Total_num
    % 读取参考图像
    path=['D:\matlab_paper\dataset\disc_initial\',num2str(j),'.jpg'];
    ref = imread(path); 

    % 读取图像
    path=['D:\matlab_paper\dataset\disc\',num2str(j),'.jpg'];
    A = imread(path);
    
    % 计算算图像的全局 SSIM 值
    % RGB图像需要指定DataFormat参数为SSC，且三个通道各有一个值
    % 灰度图像不用指定DataFormat参数的值
    ssimval = ssim(A,ref,"DataFormat","SSC"); 
    ssim_array(j) = mean(ssimval);
    disp("ssim_"+j+":"+ssim_array(j))
end
% 保存ssim:
path=['D:\matlab_paper\dataset\ssim\ssim_array.mat'];
save(path,'ssim_array');
disp("ssim保存成功！")
% 加载ssim:
load(path)
% 输出结果
disp("ssim_mean:"+mean(ssim_array));
disp("ssim_var:"+var(ssim_array));
disp("ssim_std:"+std(ssim_array));


