% TR reconstruct shepp-logan
clear;clc;
Total_num = 200; %图片数量（数据集大小）

% angle:检测视角
% sensor_num：传感器数量
angle = 2*pi;
senor_num = 100;

% angle = 3*pi/2;
% senor_num = 15;

for j = 1 : Total_num
    TR_fun(j, angle, sensor_num);
end