function [output, distance]=PPV_method(input, window_length, max_window, step, dt, v)
%% PPV-based feature extraction method (for manuscript implementation)
% -------------------------------------------------------------------------
% This function implements the PPV (Peak–Valley Product) feature extraction
% method proposed in the paper.
%
% Author      : Aoyu Sun
% Start date  : 2024.12.05
%
% INPUT:
%   input         : One-dimensional input signal
%   window_length : Window length (number of samples within each window)
%   max_window    : Maximum analysis window length (number of samples)
%   step          : Sliding step size (in samples)
%   dt            : Sampling interval
%   v             : Vehicle speed (used for distance mapping)
%
% OUTPUT:
%   output   : PPV feature sequence
%   distance : Spatial coordinate corresponding to each PPV value
% -------------------------------------------------------------------------

len=length(input);            % Length of the input signal

%% EEMD decomposition
Nstd = 0.05;                  % Ratio of added noise standard deviation to signal standard deviation
NE = 40;                      % Number of ensemble trials in EEMD
D = window_length;            % Number of samples within each sliding window
Dm = max_window;              % Maximum window length
s = step;                     % Sliding step size (in samples)

% Number of sliding windows
num = floor((len - D/2 - Dm/2) / s) + 1;

% Perform Ensemble Empirical Mode Decomposition
u = eemd(input, Nstd, NE);

% Extract the IMF6 component (low-frequency trend-related component)
lenth = length(u(1,:));
Y = u(:, 7);

%% Outlier suppression
% Method: 3-sigma rule
meanY = mean(Y);              % Mean value of the selected IMF
stdY  = std(Y);               % Standard deviation of the selected IMF
threshold = 3 * stdY;         % Outlier threshold

%% Sliding window PPV calculation
for i = 1:1:num
    % Extract signal segment within current sliding window
    An = Y((Dm/2+(i-1)*s)-D/2+1 : (Dm/2+(i-1)*s)+D/2);

    % -------------------------------
    % Peak processing
    % -------------------------------
    [pks, locs] = findpeaks(An);   %#ok<ASGLU>
    l = length(pks);
    T(i) = sum(pks);

    if l ~= 0
        % Remove abnormal peaks exceeding the threshold
        for j = 1:l
            if pks(j) > threshold
                pks(j) = 0;
            end
        end
        % Mean peak amplitude
        T(i) = sum(pks) / l;
    end

    % -------------------------------
    % Valley processing
    % -------------------------------
    [pk, loc] = findpeaks(-An);    %#ok<ASGLU>
    l = length(pk);
    B(i) = sum(pk);

    if l ~= 0
        % Remove abnormal valleys exceeding the threshold
        for j = 1:l
            if pk(j) > threshold
                pk(j) = 0;
            end
        end
        % Mean valley amplitude
        B(i) = sum(pk) / l;
    end

    % -------------------------------
    % Peak–Valley Product (PPV)
    % -------------------------------
    product_of_BT(i) = B(i) * T(i);
end

%% Output feature and spatial mapping
output = product_of_BT;

% Convert sample index to spatial distance
distance = Dm/2*dt*v : s*dt*v : (Dm/2+(num-1)*s)*dt*v;

end
