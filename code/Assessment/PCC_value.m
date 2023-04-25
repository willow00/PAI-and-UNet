%% PCC:corr()或者corrcoef()
% corr()要安装附加工具箱，在这里使用corrcoef()

clear; clc;
Total_num = 1000;
PCC_array = zeros(Total_num,1);
for j = 1 : Total_num
    % 读取参考图像
    path=['D:\matlab_paper\dataset\disc_initial\',num2str(j),'.jpg'];
    ref = imread(path); 

    % 读取图像
    path=['D:\matlab_paper\dataset\disc\',num2str(j),'.jpg'];
    A = imread(path);
    
    % 转化数据类型：可以用double()函数或im2double()函数，得到的结果相同
    ref = double(ref);
    A = double(A);

    % 计算PCC
    PCCval = corrcoef(A(:), ref(:));
    PCC_array(j) = mean(PCCval(1,2));
    disp("PCC_"+j+":"+PCC_array(j))
end
% 保存PCC:
path=['D:\matlab_paper\dataset\PCC\PCC_array.mat'];
save(path,'PCC_array');
disp("PCC保存成功！")
% 加载PCC:
load(path)
% 输出结果
disp("PCC_mean:"+mean(PCC_array));
disp("PCC_var:"+var(PCC_array));
disp("PCC_std:"+std(PCC_array));

