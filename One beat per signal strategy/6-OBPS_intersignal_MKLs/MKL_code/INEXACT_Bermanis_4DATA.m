function [output_03 ] = INEXACT_Bermanis_4DATA( Ge_s , f , F_data , x_star , e_s , gamma )

N = size(Ge_s,1);

%% step 3

% Ge_s_cross = pinv( Ge_s + 1/gamma * eye(N,N) );
% c = Ge_s_cross*f';
c = (Ge_s + (eye(N,N)/gamma) )\(f)';

%% step 4

f_s = Ge_s * c;


%% step 5
p = size(x_star,1);
l = size(f_s,1);
f_star_s = zeros(size(f_s,2),p);

for j=1:p
    tmp = repmat(x_star(j,:) , l , 1);
    tmp = sum((tmp - F_data).^2,2)';
    G_star_s = exp( -tmp / (2*(e_s)^2) );  
    
%% step 6
    f_star_s(:,j) = (G_star_s * c)';
end

output_03.f_s = f_s';
output_03.f_star_s = f_star_s;

end

