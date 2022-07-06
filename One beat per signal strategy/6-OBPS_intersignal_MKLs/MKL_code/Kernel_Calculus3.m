function tmpK = Kernel_Calculus3(Data_validation,Data_train,KernelType,K_NN)

%%% Kernel calculus to compute the affinity between all new subjects
%%% (validation or test set) and all the training set. This is preferred
%%% over Kernel_Calculus2, to speed up the computation. 

N = size(Data_train,2);

Data = [Data_train, Data_validation];

if strcmp(KernelType,'exp_l2')
    
    d = squareform(pdist(Data_train'));

    tmp = d + diag( Inf(N,1) );  %% put diagonal coefficients to -1
    tmpB = sort(tmp,1,'ascend');
    tmpB = tmpB(1:min(K_NN,N-1),:); %% approximate density from k neighbors
    tmpB = mean(tmpB,1);
    sigma = mean(tmpB(:));
    % Calculate affinity to each of the samples in the training set
    
    d = squareform(pdist(Data'));
    
    tmpK = exp( - d.^2 ./ (2*(sigma)^2) );
    
elseif strcmp(KernelType, 'ordinal')
    
    if size(Data_train,1) > 1
        % Calculate affinity between samples
        d = squareform(pdist(Data));
        d_aux = squareform(pdist(Data_train));
        range = max(d_aux(:)) - min(d_aux(:));
        tmpK = (range - d) / range;
        
    else
        
        range = max(Data) - min(Data);   
        range_aux = max(Data_train) - min(Data_train);
        tmpK = (range_aux - abs(Data - Data')) / range_aux;
        
    end
    
elseif strcmp(KernelType, 'dummy')
    
    %%% METHOD 1 
    
    % multiply by frequency; the idea is that rare categories induce higher
    % distances -> lower similarity
    freq = sum(Data)/numel(Data);
    
    % K_NN = number of categorical variables (not dummy)
    tmpK = double(Data == Data').*freq/K_NN; 
    
    
    %%% METHOD 2 
    
    % Rare events induce higher similarities, so they contribute more to
    % the OS distribution
 
%     freq1 = sum(Data == 1);
%     freq0 = sum(Data == 0);
% 
%     Data1 = repmat(Data,1,numel(Data));
%     Data2 = repmat(Data',numel(Data),1);
%     
%     if freq1 == 0 %% In case the number of 1's is 0, we avoid getting NaNs
%         Arg1 = (Data1 & Data2);
%     else
%         Arg1 = (Data1 & Data2) ./ freq1;
%     end
%     if freq0 == 0
%         Arg0 = (~Data1 & ~Data2);
%     else
%         Arg0 = (~Data1 & ~Data2) ./ freq0;
%     end
%               
% 
%     tmpK = (double(Data1 == Data2) .* Arg1) + (double(Data1 == Data2).* Arg0);
    
else
    
    tmpK = double(Data == Data');
    tmpK(Data~=Data') = 0.9;
  
  
end


% Retain only the portion of the affinity matrix that contains the
% affinities between the Validation set and the Training set
tmpK = tmpK(size(Data_train,2)+1:end,1:size(Data_train,2));