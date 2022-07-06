
alpha = 28;
alpha = string(alpha);

FEATURES = load(strcat('/Users/paubrunet/Google Drive/Documents/CBEM/TFM/Arrythmia/7-MKL-Dimensionality Reduction/MKL_seed',alpha,'_run1a.mat'));

fn = fieldnames(FEATURES);
for i = 1:numel(fn)
    if size(FEATURES.(fn{i}), 1) == 1
        FEATURES.(fn{i}) = FEATURES.(fn{i})';
    end
end

[F_data,betas,A] = MKL(struct2cell(FEATURES));
figure(2)
scatter3(F_data(:,1),F_data(:,2),F_data(:,3),'filled');
xlabel('Dim1'); ylabel('Dim2'); zlabel('Dim3'); title('Output space');
save(strcat('A', alpha,'_run1a'), 'A')
save(strcat('F_data', alpha, '_run1a'), 'F_data')
save(strcat('betas', alpha, '_run1a'), 'betas')

%load handel.mat;FE
%sound(y, 2*Fs);