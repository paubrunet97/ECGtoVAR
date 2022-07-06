clc;clear

cvx_setup

basedir = '6-MBPS_intersignal_MKLs/';
cd('6-MBPS_intersignal_MKLs/MKL_code/')
mex computeENERGY.c
mex computeSWA.c
mex computeSWB.c

% Run MKL on all the created inputs: PP' and segments.
run_names = {'P_ds1', 'QRS_ds1', 'T_ds1', 'QT_ds1', 'ST_ds1', '3seg_ds1', '6seg_ds1',
    'P_ds4', 'QRS_ds4', 'T_ds4', 'QT_ds4', 'ST_ds4', '3seg_ds4', '6seg_ds4'};

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
