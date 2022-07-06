clc;clear


addpath 'DS3 code'

% Load the dissimilarity matrices
load(strcat(basedir, 'dissimilarity_matrices.mat'))
signalnames = fieldnames(Edist);

% Lambda (alpha here) is 0.25 to pick the most representative beat per signal
% Run DS3 Subset Selection
alpha = 0.25
for i = 1:numel(signalnames)
    signalnames{i}
    D = Edist.(signalnames{i});

    p = inf; % norm used for L1/Lp optimization in DS3
    regularized = false; % true: regularized version, false: constrained version
    %alpha = 5; % regularizer coefficient
    outlierDetect = true; % if true, run the extended DS3 with outlier detection
    beta = 1000; % regularization for outlier detection
    verbose = false; % true/false: show/hide optimization steps

    CFD = ones(size(D,1),1);
    if (outlierDetect)
        D = augmentD(D,beta);
        %D = D ./ max(max(D));
        CFD = [CFD;eps];
    end

    if (regularized)
        [rho_min, rho_max] = computeRegularizer(D,p);
        options.verbose = verbose;
        options.rho = alpha * rho_max; % regularization parameter
        options.mu = 1 * 10^-1;
        options.maxIter = 3000;
        options.errThr = 1 * 10^-7;
        options.cfd = CFD;
        Z = ds3solver_regularized(D,p,options);
    else
        options.verbose = verbose;
        options.rho = alpha; % regularization parameter
        options.mu = 1 * 10^-1;
        options.maxIter = 3000;
        options.errThr = 1 * 10^-7;
        options.cfd = CFD;
        Z = ds3solver_constrained(D,p,options);
    end

    sInd = findRepresentatives(Z);
    % Save the results as a tuple, delete 1 to the indices to match python 0-indexing.
    python_indices.(signalnames{i}) = sInd - 1;

end
file1 = '_Intermediates/ds3_indices';

if alpha < 1
    file2 = string(1);
else
    file2 = string(alpha);
end

file3 = '.mat';

% Save the indices of the representative beats
save(append(file1,file2,file3), 'python_indices')
