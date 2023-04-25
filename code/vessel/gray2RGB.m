% 灰度图像转化为RGB图像
clear;clc;
addpath(genpath('D:\github_repo'));
for j = 1:1
    % 读取图像
    path = ['D:\matlab_paper\dataset\vessel_pic20\',num2str(j),'.jpg'];
    p0 = imread(path);
    p0 = im2double(p0); %由uint8转为双精度图片
    
    figure;
    set(gcf,"position",[350, 350, 330, 329]);
    imagesc(p0, [-1, 1]);
    colormap(getColorMap); %设置颜色图
    axis image;
    axis off;

    % 保存图像
    path = ['D:\matlab_paper\dataset\vessel_pic20_rgb\',num2str(j),'.jpg'];
    export_fig(path,gcf);
end