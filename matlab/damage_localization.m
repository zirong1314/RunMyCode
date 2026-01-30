%% Shape-adaptive damage localization
% x  : damage-related indicator (1D vector)
% s  : spatial coordinate (same length as x)
% k  : significance factor (recommended 2~3)

function [s_center, s_interval, mask] = damage_localization(x, s, k)

if nargin < 3
    k = 2.5; % default significance level
end

x = x(:);
s = s(:);

%% Step 1: Robust baseline estimation
mu = median(x);
sigma = mad(x, 1);  % median absolute deviation

%% Step 2: Statistical significance mask
mask = x > (mu + k * sigma);

%% Step 3 (optional): Morphological closing to connect boundary-dominated cases
mask = imclose(mask, ones(3,1));  % 1D closing

%% Step 4: Extract connected components
cc = bwconncomp(mask);

if cc.NumObjects == 0
    warning('No significant damage segment detected.');
    s_center = NaN;
    s_interval = [NaN NaN];
    return;
end

% Select the largest connected segment
seg_lengths = cellfun(@length, cc.PixelIdxList);
[~, idx] = max(seg_lengths);
damage_idx = cc.PixelIdxList{idx};

%% Step 5: Damage interval and center
s_interval = [s(damage_idx(1)), s(damage_idx(end))];
s_center   = mean(s_interval);

end
