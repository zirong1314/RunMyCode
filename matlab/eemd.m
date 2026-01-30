function allmode = eemd(Y, Nstd, NE)
% EEMD (Ensemble Empirical Mode Decomposition) implementation
%
% INPUT:
% Y    : Input signal to be decomposed (1-D data only)
% Nstd : Standard deviation ratio of the added white noise relative to Y
%        Typically Nstd = (0.1 ~ 0.4) * std(Y)
% NE   : Number of noise-added ensembles for EEMD, usually 10¨C50
%
% OUTPUT:
% allmode : An N ¡Á (m+1) matrix, where N is the length of Y and
%           m = fix(log2(N)) - 1.
%           Column 1      : original signal
%           Columns 2~m   : IMFs from high to low frequency
%           Column (m+1)  : residual (overall trend)

xsize = length(Y);
dd = 1:1:xsize; 
Ystd = std(Y);
Y = Y / Ystd;    

% Part 2: Estimate total number of IMFs and initialize output matrix
TNM = fix(log2(xsize)) - 5;   % Estimated number of IMFs
TNM2 = TNM + 2;               
for kk = 1:1:TNM2
    for ii = 1:1:xsize
        allmode(ii, kk) = 0.0;  
    end
end

% Part 3: EEMD main loop (repeat EMD NE times with different noise realizations)
for iii = 1:1:NE

    % Part 4: Add white noise to the normalized signal
    for i = 1:xsize
        temp = randn(1,1) * Nstd; % Gaussian white noise
        X1(i) = Y(i) + temp;
    end

    % Part 4: Assign original signal to the first column
    for jj = 1:1:xsize
        mode(jj,1) = Y(jj); % Original signal
    end

    % Part 5: Initialize residual signal
    xorigin = X1;   
    xend = xorigin; 

    % Part 6: IMF extraction loop
    nmode = 1;

    while nmode <= TNM
        xstart = xend;   % Signal for current sifting process
        iter = 1;        % Sifting iteration counter

        % Part 7: Perform fixed-number sifting to extract an IMF
        while iter <= 10  
            [spmax, spmin, flag] = extrema(xstart); % Detect local extrema
            upper = spline(spmax(:,1), spmax(:,2), dd); % Upper envelope
            lower = spline(spmin(:,1), spmin(:,2), dd); % Lower envelope
            mean_ul = (upper + lower) / 2;             % Mean envelope
            xstart = xstart - mean_ul;                 % Remove local mean
            iter = iter + 1;
        end

        % Part 8: Remove extracted IMF from residual signal
        xend = xend - xstart;      
        nmode = nmode + 1;              

        % Part 9: Store extracted IMF
        for jj = 1:1:xsize
            mode(jj, nmode) = xstart(jj);  
        end
    end

    % Part 10: Store final residual (overall trend)
    for jj = 1:1:xsize
        mode(jj, nmode + 1) = xend(jj);
    end

    % Accumulate ensemble results
    allmode = allmode + mode;   
end

% Part 11: Ensemble averaging and amplitude recovery
allmode = allmode / NE;    
allmode = allmode * Ystd;  
