function [error,fs,error_acc,SS,W0,fs_1] = moev(inmodel,X,MW,c,W0,indexO,Y)
    fs = sum(inmodel);
    MP=MW(inmodel,:);
    OMP = MP;
    W0([indexO],:) = 0;
    W0(inmodel,:) = OMP;    
    SS=sum(W0'.*W0');
    [~,S]=sort(SS,'descend');
    fs_new=fs;
    Xw = X(S(1:fs_new),:);
    fs_1=fs;
    [idx, ~]=kmeans(Xw',c,'maxIter',100,'replicates',20,'EmptyAction','singleton');
    result2 = ClusteringMeasure(Y,idx);
    error_acc=result2(1);
% %%%%%%
% % Calculate silhouette scores
% silhouette_scores = silhouette(Xw', idx);
% % Calculate the average silhouette score
% average_silhouette_score = mean(silhouette_scores);
% %%%%%%%
data=Xw';
k=c;
% % 
% centers = zeros(k, size(data, 2));
% for i = 1:k
%     centers(i, :) = mean(data(idx == i, :));
% end
% 
% % 
% scatter_within = zeros(k, 1);
% for i = 1:k
%     scatter_within(i) = mean(sqrt(sum((data(idx == i, :) - centers(i, :)).^2, 2)));
% end
% 
% % Davies-Bouldin index
% davies_bouldin = 0;
% for i = 1:k
%     max_ratio = 0;
%     for j = 1:k
%         if j ~= i
%             ratio = (scatter_within(i) + scatter_within(j)) / norm(centers(i, :) - centers(j, :));
%             if ratio > max_ratio
%                 max_ratio = ratio;
%             end
%         end
%     end
%     davies_bouldin = davies_bouldin + max_ratio;
% end
% davies_bouldin = davies_bouldin / k;
%%%%%%%
% center
centers = zeros(k, size(data, 2));
for i = 1:k
    centers(i, :) = mean(data(idx == i, :));
end
global_center = mean(data);
% scatter
scatter_within = 0;
for i = 1:k
    scatter_within = scatter_within + sum(sum((data(idx == i, :) - centers(i, :)).^2));
end
scatter_between = 0;
for i = 1:k
    scatter_between = scatter_between + sum(sum((centers(i, :) - global_center).^2)) * sum(idx == i);
end

% Calinski-Harabasz
calinski_harabasz = scatter_between / scatter_within * (size(data, 1) - k) / (k - 1);
error=1/calinski_harabasz;
error_acc=calinski_harabasz;
end

