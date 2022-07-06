function [x_star] = RegressionLinePointsGrid(F_data,n_dims,dim_star,n_points)


x_star = NaN(n_points,n_dims);
aux_dim = F_data(:,dim_star);
boundaries = linspace(min(aux_dim),max(aux_dim),n_points+1);

for a =1:n_points
    cond_boundaries = and(aux_dim>boundaries(a),aux_dim<boundaries(a+1));
    aux_data = F_data(cond_boundaries,1:n_dims);
    x_star(a,:) = mean(aux_data);
    
end


end

