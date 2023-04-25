clear;clc;
addpath(genpath('D:\github_repo'));
Total_num = 20;
for j = 1:Total_num

    path = ['D:\matlab_paper\last_unet\vessel_mat\',num2str(j),'.mat']
    p0 = load(path).result; % 读入tensor的矩阵3*256*256或1*256*256
    
    % 通道数为3的情况
    if size(p0,1)==3
        p = zeros(256,256,3); % 将矩阵变为256*256*3
        p(:, :, 1) = p0(1,:,:);
        p(:, :, 2) = p0(2,:,:);
        p(:, :, 3) = p0(3,:,:);
    end

    % 通道数为1的情况
    if size(p0,1)==1
        p = squeeze(p0); % 将矩阵变为256*256
    end

    %由矩阵画图，保存
    figure;
    set(gcf,"position",[400, 400, 330, 329]);
    imagesc(p, [-1, 1]);
    colormap(getColorMap);
    % colorbar;
    axis image;
    axis off;
    path = ['D:\matlab_paper\last_unet\vessel_rs\',num2str(j),'.jpg']
    export_fig(path,gcf);
    close all;
end

