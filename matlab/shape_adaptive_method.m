clear all;
clc;

baseFileName = { 'ESY-5'};     % Base name of the input data file
fileType = '.txt';            % File extension
numFiles = 1;                 % Number of data files (total datasets: 36)

for t = 1:numFiles
    %% Construct full file name and read data
    filename = [char(baseFileName(t)), fileType];
    M = readmatrix(filename);

    % Extract signal segment used for analysis
    input = M(2189:5900, 2);

    %% PS computation under multiple scales
    j = 1;                    % Scale index

    dt = 0.0001;              % Sampling interval (≈ 9.7 mm per sample)
    v  = 350 / 3.6;           % Vehicle speed (m/s)

    max_window = 240;         % Maximum window length (samples)
    min_window = 120;         % Minimum window length (samples) —— critical parameter
    step = 5;                 % Sliding step (5 samples ≈ 50 mm)
    interval = 5;             % Window scale increment

    % Loop over window scales
    for i = min_window : interval : max_window
        % PPV feature extraction at current scale
        % Input arguments:
        %   input       : input signal
        %   i           : window length
        %   max_window  : maximum window length
        %   step        : sliding step
        %   dt          : sampling interval
        %   v           : vehicle speed
        [y, x] = PPV_method(input, i, max_window, step, dt, v);

        len = length(y);

        % Store PPV values and corresponding spatial positions
        P(1:len, j)    = y;   % PPV values at current scale
        dist(1:len, j) = x;   % Spatial center positions for PPV values

        j = j + 1;
    end
end

%% Multi-scale fusion of PPV spectra
% Sum PPV values across all window scales to obtain a fused spectral sequence
x = sum(P, 2);                % Fused PPV spectral sequence
s = dist(:, 1);               % Corresponding spatial coordinate (distance)

%% Damage localization based on fused spectrum
% damage_localization returns:
%   s_center   : estimated damage center position
%   s_interval : estimated damage spatial interval
%   mask       : logical mask indicating detected damage-related samples
[s_center, s_interval, mask] = damage_localization(x, s, 2.5);

%% Visualization of damage localization results
figure;
plot(s, x, 'k'); hold on;             % Fused PPV spectrum
plot(s(mask), x(mask), 'ro');         % Detected damage-related points

% Estimated damage boundaries
xline(s_interval(1), '--b', 'Left boundary');
xline(s_interval(2), '--b', 'Right boundary');

xlabel('Distance (m)');
ylabel('Spectral sequence value (m^2/s^4)');
xlim([min(s) max(s)]);
grid on;
