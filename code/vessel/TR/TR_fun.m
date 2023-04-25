function[] = TR_fun(j, angle, sensor_num) %j:计数
addpath(genpath('D:\github_repo'));
p0_magnitude = 2;
path = ['D:\matlab_paper\dataset\vessel_pic20\',num2str(j),'.jpg']
p0 = imread(path);
p0 = im2double(p0); %由uint8转为双精度图片
p0 = p0_magnitude*p0;

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
kgrid.makeTime(medium.sound_speed);

% set the input options
input_args = {'Smooth', false,'PMLSize', PML_size, 'PMLInside', false, 'PlotPML', false};

% run the simulation
sensor_data = kspaceFirstOrder2D(kgrid, medium, source, sensor, input_args{:});

% add noise to the recorded sensor data
signal_to_noise_ratio = 30;	% [dB]
% addNoise Add Gaussian noise to a signal for a given SNR.
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

%二进制掩码(优点见官网TR_circle例子)
% create a binary sensor mask of an equivalent continuous circle 
sensor_radius_grid_points = round(sensor_radius / kgrid_recon.dx);
binary_sensor_mask = makeCircle(kgrid_recon.Nx, kgrid_recon.Ny, kgrid_recon.Nx/2 + 1, kgrid_recon.Ny/2 + 1, sensor_radius_grid_points, sensor_angle);

% assign to sensor structure
sensor.mask = binary_sensor_mask;

% interpolate data to remove the gaps and assign to sensor structure
%interpCartData Interpolate data from a Cartesian to a binary sensor mask.
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
path = ['D:\matlab_paper\vessel\pressure_sensor\',num2str(j),'.jpg'];
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
path = ['D:\matlab_paper\shepp_logan\shepp_logan_initial\',num2str(j),'.jpg']
export_fig(path,gcf);

% plot the simulated sensor data
figure;
set(gcf,"position",[400, 400, 330, 329]);
imagesc(sensor_data, [-1, 1]);
colormap(getColorMap);
ylabel('Sensor Position');
xlabel('Time Step');
colorbar;
path = ['D:\matlab_paper\vessel\sensor_data\',num2str(j),'.jpg'];
export_fig(path,gcf);

% plot the reconstructed initial pressure using the interpolated data
figure;
set(gcf,"position",[400, 400, 330, 329]);
imagesc(kgrid_recon.y_vec * 1e3, kgrid_recon.x_vec * 1e3, p0_recon_interp, [-1, 1]);
colormap(getColorMap);
% ylabel('x-position [mm]');
% xlabel('y-position [mm]');
axis image;
axis off;
path = ['D:\matlab_paper\vessel\vessel_rs\',num2str(j),'.jpg'];
export_fig(path,gcf);

close all; % 关闭所有figure
end