function disc = makeDisc_2(Nx, Ny, cx, cy, radius, plot_disc)
% define literals
MAGNITUDE = rand; % 控制圆盘的颜色

% check for plot_disc input
% nargin: 函数输入参数数目
if nargin < 6
    plot_disc = false;
end

% force integer values for grid size(强制网格大小的整数值)
Nx = round(Nx);
Ny = round(Ny);
cx = round(cx);
cy = round(cy);

% check for zero values
% Y = floor(X) 向下取整
if cx == 0
    cx = floor(Nx/2) + 1;
end
if cy == 0
    cy = floor(Ny/2) + 1;
end

% check the inputs
if cx < 1 || cx > Nx || cy < 1 || cy > Ny
    error('Disc center must be within grid.');
end

% create empty matrix
disc = zeros(Nx, Ny);

% define pixel map
r = makePixelMap(Nx, Ny, 'Shift', [0, 0]);

% create disc
disc(r <= radius) = MAGNITUDE;

% shift centre
%Y = ceil(X) 向上取整
cx = round(cx) - ceil(Nx/2);
cy = round(cy) - ceil(Ny/2);
% circshift(disc, [cx, cy])将disc沿x轴循环平移cx个单位，沿y轴循环平移cy个单位，
disc = circshift(disc, [cx, cy]);

% create the figure
if plot_disc
    figure;
    %imagesc(___,clims) 指定映射到颜色图的第一个和最后一个元素的数据值。
    % 将 clims 指定为 [cmin cmax] 形式的二元素向量，
    % 其中小于或等于 cmin 的值映射到颜色图中的第一种颜色，
    % 大于或等于 cmax 的值映射到颜色图中的最后一种颜色。在名称-值对组参数后指定 clims。
    imagesc(disc, [0, 1]);
    % colormap:查看并设置当前颜色图
    % getColorMap Return default k-Wave color map
    % getColorMap返回用于在k-Wave工具箱中显示和可视化的默认颜色映射。
    % 零值显示为白色，正值显示为黄色至红色至黑色，负值显示为浅蓝色至深蓝色灰色。
    % 如果没有提供num_colors的值，cm将具有256种颜色。
    colormap(getColorMap);
    % axis tight 将坐标轴显示的框调整到显示数据最紧凑的情况，也就根据x，y坐标的最大值和最小值最紧凑调整坐标轴的显示范围；
    % axis equal 等比例显示x，y坐标轴，由于x，y轴的范围是可以分辨调整的，所以很容易让得到的图像在屏幕上显示，x，y方向的比例不一致，圆形显示为椭圆形；为了方便比较，这个命令可以让x轴和y轴比例一致，
    % 但是分别执行以上两个命令，会互相覆盖，紧凑显示的时候，比例不对，比例对了的时候，显示不紧凑，留太多空白；
    % axis image，相当于以上两个命令的合体，能够同时实现紧凑以及xy比例一致两个功能。
    axis image;

    xlabel('y-position [grid points]');
    ylabel('x-position [grid points]');
end