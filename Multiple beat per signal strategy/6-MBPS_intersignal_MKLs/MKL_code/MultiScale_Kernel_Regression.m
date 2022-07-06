function [F_star_out] = Plot_Variability_descriptor(Features,F_data,ellipse_dim,Opt_gamma)

% Same code as Plot

it_max = 50;
% P = 10;
P = ellipse_dim;

n_features = size(Features,1);
F_star_out = {};
dst = squareform(pdist(F_data(:,1:P)));
diam = max(dst(:));
T = diam;
dst = dst + diag( repmat(Inf,size(F_data,1),1) );  %% put diagonal coefficients to -1
dst = min(dst , [] , 1);  %% find closest neighbour
density = mean(dst(:));

for n = 1 : n_features
    % MSE Algorithm
    s = 0;
    f = Features{n}.Feature;
    
    F_data = F_data(:,1:P);
    
    F_data_mean = mean(F_data);
    x_star = zeros(ellipse_dim*5,P);
    for i = 1:ellipse_dim
        dev = 2*std(F_data(:,i));
%         dev = 1.5*std(F_data(:,i));
        val = [-dev,-0.5*dev,0,0.5*dev,dev];
        for j = 1:length(val)
            x_star(5*(i-1)+j,:) = F_data_mean;
            x_star(5*(i-1)+j,i) = x_star(5*(i-1)+j,i) + val(j);
            
        end
    end
    
    F_s_old = zeros(size(f));
    F_star_s_old = zeros(size(f,1),ellipse_dim*5);
    
    while ( ( s <= it_max ) && ( (T/2^s) > (2*density) ) )
        % Kernel bandwidth
        e_s = T / 2^s;
        dst = squareform(pdist(F_data));
        Ge_s = exp( -dst.^2 / (2*(e_s)^2) );
        output_03 = INEXACT_Bermanis_4DATA( Ge_s , f - F_s_old , F_data , x_star , e_s , Opt_gamma(n) );
        F_s = F_s_old + output_03.f_s;
        F_star_s = F_star_s_old + output_03.f_star_s;
        s = s+1;
        F_s_old = F_s;
        F_star_s_old = F_star_s;
    end
    
    F = F_s_old;
    F_star = F_star_s_old;
    F_star_out {n}= F_star;
end

end




