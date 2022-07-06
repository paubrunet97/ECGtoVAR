function [ K,sigma ] = Kernel_Calculus( Data,KernelType,K_NN,alpha )

% ================================================================================
%  This Unsupervised Multiple Kernel Learning (U-MKL) package is (c) BCNMedTech
%  UNIVERSITAT POMPEU FABRA
% 
%  This U-MKL package is free software: you can redistribute it and/or modify
%  it under the terms of the GNU Affero General Public License as
%  published by the Free Software Foundation, either version 3 of the
%  License, or (at your option) any later version.
% 
%  This U-MKL package is distributed in the hope that it will be useful,
%  but WITHOUT ANY WARRANTY; without even the implied warranty of
%  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%  GNU Affero General Public License for more details.
% 
%  You should have received a copy of the GNU Affero General Public License
%  along with this program.  If not, see <http://www.gnu.org/licenses/>.
% 
%  Author:
%  Sergio Sanchez-Martinez 
%
%  Contributors: 
%  Nicolas Duchateau
%  Gemma Piella
%  Constantine Butakoff
% ================================================================================

%%% If the MATLAB VERSION IS OLDER THAN 2017b, the comparison Data == Data'
%%% may fail. In order to avoid this, uncomment the following line of code
%Data = repmat(Data,1,size(Data,1));

    
if strcmp(KernelType,'exp_l2')
    
    d = squareform(pdist(Data));
    N = size(Data,1);
    
    tmp = d + diag( Inf(N,1) );  %% put diagonal coefficients to -1
    tmpB = sort(tmp,1,'ascend');
    tmpB = tmpB(1:min(K_NN,N-1),:); %% approximate density from k neighbors
    tmpB = mean(tmpB,1);
    sigma = mean(tmpB(:));
    K = exp( -alpha * ( d.^2 / (2*(sigma)^2) ) );
      
elseif strcmp(KernelType,'exp_l2_density')
    
    d = squareform(pdist(Data));
    N = size(Data,1);

    tmp = d + diag( Inf(N,1) );  %% put diagonal coefficients to -1
    tmpB = sort(tmp,1,'ascend');
    tmpB = tmpB(1:min(K_NN,N-1),:); %% approximate density from k neighbors
    sigma = mean(tmpB,1);
    
    sigma(sigma==0) = min(sigma(sigma>0));
    sigma = repmat(sigma,length(sigma),1)';
    
    K = exp( -alpha * ( d.^2 ./ (2*(sigma)^2) ) );
    
elseif strcmp(KernelType, 'prob_cat')
    
    % Probabilistic kernel for categorical data
    % (https://stats.stackexchange.com/questions/222344/same-kernel-for-mixed-categorical-data)
    K = Data == Data';
    
    bins = unique(Data);
    bin_counts = histc(Data,bins);
    
    prob = zeros(1,length(Data));
    for i = 1:numel(bins)
        prob(Data==bins(i)) = bin_counts(i)/length(Data); 
    end
    
    prob = repmat(prob,length(Data),1); 
    
    alpha = 1;
    
    K = K.* (1 - prob.*alpha).^(1/alpha); 
    if min(prob(:)) > 0.05
        K(K==0) = min(prob(:)) - 0.05;
    end
    
elseif strcmp(KernelType, 'ordinal')
    
    if size(Data,2) > 1
        
        d = squareform(pdist(Data));
        range = max(d(:)) - min(d(:));
        K = (range - d) / range;
        
    else
    
        range = max(Data) - min(Data);    
        K = (range - abs(Data - Data')) / range;
        
    end
    
elseif strcmp(KernelType, 'Longitudinal_Gaussian')
    n_timepoints = K_NN(2);
    N = size(Data,1);
    N_tp = N/n_timepoints;
    K_intraslice = zeros(N);
    mask_interslice = zeros(N);
    for aux_tp = 1:n_timepoints
        d = squareform(pdist(Data((aux_tp-1)*N_tp+1:N_tp*aux_tp,:)));
        tmp = d + diag( Inf(N_tp,1) );  %% put diagonal coefficients to -1
        tmpB = sort(tmp,1,'ascend');
        tmpB = tmpB(1:min(K_NN(1),N_tp-1),:); %% approximate density from k neighbors
        tmpB = mean(tmpB,1);
        sigma = mean(tmpB(:));
        K_intraslice((aux_tp-1)*N_tp+1:N_tp*aux_tp,(aux_tp-1)*N_tp+1:N_tp*aux_tp) = exp( -alpha * ( d.^2 / (2*(sigma)^2) ) );
        if aux_tp>1
            mask_interslice = mask_interslice + diag(ones(N-abs(N_tp*(aux_tp-1)+1),1),N_tp*(aux_tp-1)+1);
            mask_interslice = mask_interslice + diag(ones(N-abs(N_tp*(aux_tp-1)+1),1),-N_tp*(aux_tp-1)-1);
        end
    end
    
    d = squareform(pdist(Data));
    tmp = d + diag( Inf(N,1) );  %% put diagonal coefficients to -1
    tmpB = sort(tmp,1,'ascend');
    tmpB = tmpB(1:min(K_NN(1),N-1),:); %% approximate density from k neighbors
    tmpB = mean(tmpB,1);
    sigma = mean(tmpB(:));
    K_interslice = exp( -alpha * ( d.^2 / (2*(sigma)^2) ) ).* mask_interslice;
        
    K = K_intraslice + K_interslice;
else
    
    K = double(Data == Data');
    K(Data~=Data') = 0.9;
  
end



