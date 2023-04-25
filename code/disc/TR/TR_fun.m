% 时间反演算法（TR）重建光声图像
% 理想条件视角：2pi，传感器数量：100
% 非理想条件视角：3pi/2，传感器数量：15
function [] = TR_fun(j, angle, sensor_num) 

% 导入export_fig库：A MATLAB toolbox for exporting publication quality figures
% export_fig库 github链接：https://github.com/altmany/export_fig
% export_fig库下载地址：https://ww2.mathworks.cn/matlabcentral/fileexchange/23629-export_fig
addpath(genpath('D:\github_repo'));
Nx = 216;
Ny = 216;

% create initial pressure distribution using makeDisc
disc_num = randi(3)+2; %随机：一幅图像中有3-5个圆盘
disc_radius = randi(20, disc_num, 1)+5; %半径:5-25
disc_x_pos = randi(86, disc_num , 1)+65; %圆心x:65-151
disc_y_pos = randi(86, disc_num , 1)+65; %圆心y:65-151

p0 = zeros(Nx, Ny);

for i = 1: disc_num 
    disc_magnitude = 2;
    disc = disc_magnitude * makeDisc_2(Nx, Ny, disc_x_pos(i), disc_y_pos(i), disc_radius(i));
    p0 = p0 + disc;
end

% 将初始压力p0(smooth前)的保存下来，方便再次加载使用
path=['D:\matlab_paper\disc_phantom\disc_p0\',num2str(j),'.mat'];
save(path,'p0');
% 加载使用方法：
% path=['D:\matlab_paper\disc_phantom\disc_p0\',num2str(j),'.mat'];
% load(path);

% 画出圆盘图像并保存
figure;
set(gcf,"position",[400, 400, 330, 329]); 
imagesc(p0, [-1, 1]);
colormap(getColorMap);
% colorbar;
axis image;
axis off;
path=['D:\matlab_paper\disc_phantom\disc_pic\',num2str(j),'.jpg'];
export_fig(path,gcf);

% 直接用上述p0作为初始压力，不另外加载图片
% assign the grid size and create the computational grid
PML_size = 20;              % size of the PML in grid points
Nx = 256 - 2 * PML_size;    % number of grid points in the x direction
Ny = 256 - 2 * PML_size;    % number of grid points in the y direction
x = 10e-3;                  % total grid size [m]
y = 10e-3;                  % total grid size [m]
dx = x / Nx;                % grid point spacing in the x direction [m]
dy = y / Ny;                % grid point spacing in the y direction [m]
kgrid = kWaveGrid(Nx, dx, Ny, dy);

% resize the input image to the desired number of grid points
p0 = resize(p0, [Nx, Ny]);

% smooth the initial pressure distribution and restore the magnitude
p0 = smooth(p0, true);

% assign to the source structure
source.p0 = p0;

% define the properties of the propagation medium
medium.sound_speed = 1500;  % [m/s]

% define a centered Cartesian circular sensor
sensor_radius = 4.5e-3;     % [m]
sensor_angle = angle;      % [rad]
sensor_pos = [0, 0];        % [m]
num_sensor_points = sensor_num; 
cart_sensor_mask = makeCartCircle(sensor_radius, num_sensor_points, sensor_pos, sensor_angle);

% assign to sensor structure
sensor.mask = cart_sensor_mask;

% create the time array
% The time-step dt is chosen based on the CFL number (the default is 0.3), 
% where dt = cfl * dx / sound_speed.
% 库朗数就是在一个时间步长里一个流体质点可以穿过多少个网格。显然，时间步长越大库朗数越大
% Nt是sqrt(216^2+216^2)/0.3向上取整：指声波传过对角线所需时间间隔数量。
kgrid.makeTime(medium.sound_speed);

% set the input options
input_args = {'Smooth', false,'PMLSize', PML_size, 'PMLInside', false, 'PlotPML', false};

% run the simulation
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

% add noise to the recorded sensor data
signal_to_noise_ratio = 30;	% [dB]
% addNoise Add Gaussian noise to a signal for a given SNR.
% 信噪比是指一个电子设备或者电子系统中信号与噪声的比例
sensor_data = addNoise(sensor_data, signal_to_noise_ratio, 'peak');

% create a second computation grid for the reconstruction to avoid the
% inverse crime
Nx = 300;           % number of grid points in the x direction
Ny = 300;           % number of grid points in the y direction
dx = x/Nx;          % grid point spacing in the x direction [m]
dy = y/Ny;          % grid point spacing in the y direction [m]
kgrid_recon = kWaveGrid(Nx, dx, Ny, dy);

% use the same time array for the reconstruction
kgrid_recon.setTime(kgrid.Nt, kgrid.dt); 

% reset the initial pressure
source.p0 = 0;

% % assign the time reversal data
% sensor.time_reversal_boundary_data = sensor_data;
% 
% % run the time-reversal reconstruction
% p0_recon = kspaceFirstOrder2D(kgrid_recon, medium, source, sensor, input_args{:});

% 二进制掩码(优点见官网TR_circle例子)
% create a binary sensor mask of an equivalent continuous circle 
sensor_radius_grid_points = round(sensor_radius / kgrid_recon.dx);
binary_sensor_mask = makeCircle(kgrid_recon.Nx, kgrid_recon.Ny, kgrid_recon.Nx/2 + 1, kgrid_recon.Ny/2 + 1, sensor_radius_grid_points, sensor_angle);

% assign to sensor structure
sensor.mask = binary_sensor_mask;

% interpolate data to remove the gaps and assign to sensor structure
% interpCartData Interpolate data from a Cartesian to a binary sensor mask.
sensor.time_reversal_boundary_data = interpCartData(kgrid_recon, sensor_data, cart_sensor_mask, binary_sensor_mask);

% run the time-reversal reconstruction
p0_recon_interp = kspaceFirstOrder2D(kgrid_recon, medium, source, sensor, input_args{:});

% =========================================================================
% VISUALISATION
% =========================================================================

% plot the initial pressure and sensor distribution
figure;
set(gcf,"position",[400, 400, 330, 329]);
imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, p0 + cart2grid(kgrid, cart_sensor_mask), [-1, 1]);
colormap(getColorMap);
ylabel('x-position [mm]');
xlabel('y-position [mm]');
axis image;
path=['D:\matlab_paper\disc_phantom\pressure_sensor\',num2str(j),'.jpg'];
export_fig(path,gcf);


% plot the initial pressure
figure;
set(gcf,"position",[400, 400, 330, 329]);
imagesc(kgrid.y_vec * 1e3, kgrid.x_vec * 1e3, p0, [-1, 1]);
colormap(getColorMap);
% ylabel('x-position [mm]');
% xlabel('y-position [mm]');
axis image;
axis off;
path=['D:\matlab_paper\disc_phantom\disc_initial\',num2str(j),'.jpg'];
export_fig(path,gcf);


% plot the simulated sensor data
figure;
set(gcf,"position",[400, 400, 330, 329]);
imagesc(sensor_data, [-1, 1]);
colormap(getColorMap);
ylabel('Sensor Position');
xlabel('Time Step');
colorbar;
path=['D:\matlab_paper\disc_phantom\sensor_data\',num2str(j),'.jpg'];
export_fig(path,gcf);


% % plot the reconstructed initial pressure 
% figure;
% set(gcf,"position",[400, 400, 330, 329]);
% imagesc(kgrid_recon.y_vec * 1e3, kgrid_recon.x_vec * 1e3, p0_recon, [-1, 1]);
% colormap(getColorMap);
% % ylabel('x-position [mm]');
% % xlabel('y-position [mm]');
% axis image;
% axis off;
% path=['D:\matlab_paper\disc_phantom\disc_tr1\',num2str(j),'.jpg'];
% export_fig(path,gcf);


% plot the reconstructed initial pressure using the interpolated data
figure;
set(gcf,"position",[400, 400, 330, 329]);
imagesc(kgrid_recon.y_vec * 1e3, kgrid_recon.x_vec * 1e3, p0_recon_interp, [-1, 1]);
colormap(getColorMap);
% ylabel('x-position [mm]');
% xlabel('y-position [mm]');
axis image;
axis off;
path=['D:\matlab_paper\disc_phantom\disc_tr2\',num2str(j),'.jpg'];
export_fig(path,gcf);

close all; %关闭所有figure
end



