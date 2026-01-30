%% 多尺度融合分析（Multi-scale fusion）
% -------------------------------------------------------------------------
% This script performs multi-scale fusion analysis based on the PPV method.
% PPV features are calculated under multiple window scales and visualized
% as a 2D peak–valley spectrum.
% -------------------------------------------------------------------------

clear all;
clc;
tic

baseFileName = { 'ESX-5'};     % Base name(s) of input data file(s)
fileType = '.txt';            % File extension
numFiles = 1;                 % Number of data files (total datasets: 36)

for t = 1:numFiles
    %% Construct full file name and read data
    filename = [char(baseFileName(t)), fileType];
    M = readmatrix(filename);

    % Extract signal segment used for analysis
    input = M(2189:5900, 2);

    %% Peak–Valley Spectrum (PVS) calculation
    j = 1;                    % Scale index

    dt = 0.0001;              % Sampling interval (corresponding to ~9.7 mm per sample)
    v  = 350 / 3.6;           % Vehicle speed (m/s)

    max_window = 240;         % Maximum window length (samples)
    min_window = 120;         % Minimum window length (samples) —— critical parameter
    step = 5;                 % Sliding step size (5 samples ≈ 50 mm)
    interval = 5;             % Window scale increment

    % Loop over multiple window scales
    for i = min_window : interval : max_window
        % PPV feature extraction at current scale
        % Input arguments:
        %   input       : signal
        %   i           : window length
        %   max_window  : maximum window length
        %   step        : sliding step size
        %   dt          : sampling interval
        %   v           : vehicle speed
        [y, x] = PPV_method(input, i, max_window, step, dt, v);

        len = length(y);

        % Store PPV results and corresponding spatial locations
        P(1:len, j)    = y;   % PPV values at current scale
        dist(1:len, j) = x;   % Spatial positions corresponding to PPV values

        j = j + 1;
    end

    %% Visualization: multi-scale PPV spectrum
    figure

    x_axis = dist(:,1);                              % Spatial distance (m)
    y_axis = 1 : length(min_window:step:max_window); % Window scale index

    h = imagesc(x_axis, y_axis, P');              
    axis xy
    colorbar;

    % Color axis range (empirically determined)
    caxis([-30, 80]);

    xlabel('Distance (m)', ...
        'fontsize',20,'FontName','Times New Roman','Fontweight','bold');
    ylabel('Window scale', ...
        'fontsize',20,'FontName','Times New Roman','Fontweight','bold');
end
toc