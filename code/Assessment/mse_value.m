%% MSE:immse()函数
clear; clc;
Total_num = 1000; % 图片数量
mse_array = zeros(Total_num,1);

for j = 1 : Total_num
    % 读取参考图像
    path=['D:\matlab_paper\dataset\disc_initial\',num2str(j),'.jpg'];
    ref = imread(path); 

    % 读取图像
    path=['D:\matlab_paper\dataset\disc\',num2str(j),'.jpg'];
    A = imread(path);

    % 计算mse(图像的数据类型为uint8，0-255)
    mse_val = immse(ref, A);
    disp("mse_"+j+":"+mse_val)
    mse_array(j) = mse_val;
    
end

% 保存mse:
path=['D:\matlab_paper\dataset\mse\mse_array.mat'];
save(path,'mse_array');
disp("mse保存成功！")
% 加载mse_good:
load(path)
% 输出：
disp("平均mse: "+mean(mse_array))
disp("mse的方差: "+var(mse_array))
disp("mse的标准差: "+std(mse_array))
