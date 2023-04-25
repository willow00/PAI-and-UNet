%% psnr:(可同时返回snr)
clear; clc;
Total_num = 1000;
psnr_array = zeros(Total_num,1);
for j = 1 : Total_num
    % 读取参考图像
    path=['D:\matlab_paper\dataset\disc_initial\',num2str(j),'.jpg'];
    ref = imread(path); 

    % 读取图像
    path=['D:\matlab_paper\dataset\disc\',num2str(j),'.jpg'];
    A = imread(path);

    % 计算psnr值：(以及snr值)
    % RGB图像需要指定DataFormat参数为SSC
    % 灰度图像不用指定DataFormat参数的值
    %[peaksnr, snr_val] = psnr(A, ref, 'DataFormat', 'SSC');
    peaksnr = psnr(A, ref, 'DataFormat', 'SSC');
    psnr_array(j) = peaksnr;
    disp("psnr_"+j+":"+peaksnr)
end

% 保存psnr:
path=['D:\matlab_paper\dataset\psnr\psnr_array.mat'];
save(path,'psnr_array');
disp("psnr保存成功！")
% 加载psnr:
load(path)
% 输出结果
disp("psnr_mean:"+mean(psnr_array));
disp("psnr_var:"+var(psnr_array));
disp("psnr_std:"+std(psnr_array));




