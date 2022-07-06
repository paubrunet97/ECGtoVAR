clc;clear

cvx_setup

basedir = '6-OBPS_intersignal_MKLs/';
cd('6-OBPS_intersignal_MKLs/MKL_code/')
mex computeENERGY.c
mex computeSWA.c
mex computeSWB.c

% Run MKL on all the created inputs: PP' and segments.
run_names = {'1seg', '3seg', 'P', 'QRS', 'ST', 'QT', 'T'}

for run = 1:length(run_names)

    FEATURES = load(strcat(basedir, 'MKLinput_', run_names{run}, '.mat'));

    fn = fieldnames(FEATURES);
    for i = 1:numel(fn)
        if size(FEATURES.(fn{i}), 1) == 1
            FEATURES.(fn{i}) = FEATURES.(fn{i})';
        end
    end

    FEATURES = struct2cell(FEATURES);

    for i=1:numel(FEATURES)
        options.Kernel{i}.KernelType = 'exp_l2';
        options.Kernel{i}.Parameters = floor(sqrt(size(FEATURES{1},1)));
    end
    options.NumberOfIterations = 250;

    [MKLoutput,betas,A] = MKL(FEATURES, options);

    save(strcat(basedir, 'MKLoutput_', run_names{run}, '.mat'))
end
