load('../../dataset/cvpr_feat.mat');


mex dist2aff.c  ;
mex evrot.c  ;
mex scale_dist.c ;

% parameter
neighbor_num = 15;
% scale = 0.04;
CLUSTER_NUM_CHOISES = [5:40];

% normalize
X = Feature(601:800,:);
[n,p] = size(X);
X = X - repmat(mean(X,1),n,1);
X = X/max(max(abs(X)));

D = dist2(X,X);
% A = exp(-D/(scale^2));
[~,A_LS,~] = scale_dist(D, floor(neighbor_num/2));

ZERO_DIAG = ~eye(size(X,1));
% A = A.*ZERO_DIAG;
A_LS = A_LS.*ZERO_DIAG;

[clusters_RLS, rlsBestGroupIndex, qualityRLS] = cluster_rotate(A_LS, CLUSTER_NUM_CHOISES,0,1);
% [clusters_R, rBestGroupIndex, qualityR] = cluster_rotate(A, CLUSTER_NUM_CHOISES,0,1)
