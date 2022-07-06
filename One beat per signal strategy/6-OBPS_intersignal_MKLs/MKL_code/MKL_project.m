function F_data_tst = MKL_project(FEATURES_tst,FEATURES_trn,options,A,betas,N_dims)

N_trn = size(FEATURES_trn{1,1},2);
N_tst = size(FEATURES_tst{1,1},2);
n_features = numel(FEATURES_trn);

opt_Kernel = options.Kernel;
Kernel = zeros(N_tst,N_trn);
for c=1:n_features

    tmpKernel = Kernel_Calculus3(FEATURES_tst{c}, ...
                                FEATURES_trn{c},...
                                opt_Kernel{c}.KernelType, ...
                                opt_Kernel{c}.Parameters);
    Kernel = Kernel + tmpKernel*betas(c);

end

%%% Data projected in the learned representation space
F_data_tst = (Kernel*A(:,1:N_dims));


