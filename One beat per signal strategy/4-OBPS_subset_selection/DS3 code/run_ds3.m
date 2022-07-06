clc; clear

denoised = false;
basedir = '/Users/paubrunet/Google Drive/Documents/TFM/LongQT/5-DS3-Subset Selection/';
if denoised
    mats = load(strcat(basedir, 'x-dissimilarity_matrices_d.mat'));
else
    mats = load(strcat(basedir, 'x-dissimilarity_matrices.mat'));
end

signalnames = fieldnames(mats);

for alpha = [0.25, 6]
    numInd = 0;
    for i = 1:numel(signalnames)
        
        D = mats.(signalnames{i});
        
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
        
        
        % run DS3
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
        numInd = numInd + size(sInd, 2);
        % save the results as a tuple, delete 1 to the indices to match python
        % 0-indexing.
        python_indices.(signalnames{i}) = sInd - 1;
        
    end
    %numInd_matrix(alpha,1) = alpha;
    %numInd_matrix(alpha,2) = numInd;
    file1 = 'ds3_indices';
    
    if alpha < 1
        file2 = string(1);
    else
        file2 = string(alpha);
    end
    
    if denoised
        file3 = '_d.mat';
    else
        file3 = '.mat';
    end
    
    filename = append(basedir, file1,file2,file3)
    save(filename, 'python_indices')
end


