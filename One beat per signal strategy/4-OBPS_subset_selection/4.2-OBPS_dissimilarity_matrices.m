clc; clear
cvx_setup

cd('../7-intersignalMKL/MKL_batch code')
mex computeENERGY.c
mex computeSWA.c
mex computeSWB.c

basedir = '_Intermediates/';

% Load the signal acceptable beats MKL inputs
FileDirs = struct2cell(dir(basedir));
FileNames = cellfun(@(x) x(1:end-4), FileDirs(1,:), 'UniformOutput', false);
FileNames = FileNames(~cellfun('isempty',FileNames));

% Run MKL for each signal acceptable beats
for signal = 2:length(FileNames)
    FEATURES = load(strcat(basedir, FileNames{signal}));

    fn = fieldnames(FEATURES);
    for i = 1:numel(fn)
        if size(FEATURES.(fn{i}), 1) == 1
            FEATURES.(fn{i}) = FEATURES.(fn{i})';
        end
        if var(FEATURES.(fn{i})) == 0
            FEATURES = rmfield(FEATURES , fn{i});
        end
    end
    FEATURES = struct2cell(FEATURES);
        
    for j = 1:numel(FEATURES)
        options.Kernel{j}.KernelType = 'exp_l2';
        options.Kernel{j}.Parameters = size(FEATURES{1},1);
    end
    options.NumberOfIterations = 250;
    options.AffinityNN = size(FEATURES{1},1);

    [MKLoutput,~,~] = MKL(FEATURES, options);

    % Compute the euclidean distance between MKLoutput pairs (first 3 dims), save them all in a struct
    if size(FEATURES{1},1) >3
        X = MKLoutput(:,1:3);
    else
        X = MKLoutput;
    end
    edist = squeeze(sqrt(sum(bsxfun(@minus,X,reshape(X',1,size(X,2),size(X,1))).^2,2)));
    Edist.(strrep(FileNames{signal},'#','z')) = edist;
end

save(append(basedir, 'dissimilarity_matrices.mat'), "Edist")
