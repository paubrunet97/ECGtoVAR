function [x_star] = RegressionLinePoints(F_data,n_dims,n_points,dim2sort)


cond_reclust = 1;
while cond_reclust ~= 0
    aux_count = 1;
    clust2del = [];
    [idx_cluster,~] = kmeans(F_data(:,1:n_dims), n_points,'Replicates',50);
    uClust = unique(idx_cluster);
    for a = 1:length(uClust)
%         disp(['Cluster ',num2str(a),': ',num2str(sum(idx_cluster==a))])
        
        if sum(idx_cluster==a)<2
            clust2del(aux_count) = a;
            aux_count = aux_count + 1;
        end
    end
    if isempty(clust2del)
        cond_reclust = 0;
    else
        for aux_del =1:length(clust2del)
            F_data(idx_cluster==clust2del(aux_del),:) = [];
            idx_cluster(idx_cluster==clust2del(aux_del)) = [];
        end
    end
%     disp('======================')
end

for a = 1:n_points
    x_star(a,:) = mean(F_data(idx_cluster==a,1:n_dims));
end

x_star = sortrows(x_star,dim2sort);


end

