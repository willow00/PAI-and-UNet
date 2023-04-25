% create shepp-logan
clear;clc;
addpath(genpath('D:\github_repo'));
P = phantom();
imshow(P)
axis off;
set(gcf,"Units","Pixels","Position",[200, 200, 463, 460]); 
export_fig("sl.jpg",gcf); % 保存图片
