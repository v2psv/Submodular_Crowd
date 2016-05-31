% Computes the Gaussian mutual information between a set and its complement
%
% function mi = sub_fn_mi(sigma,V) 
% sigma: Covariance Matrix
% set: the ground set
%
% Example: F = sub_fn_mi(0.5*eye(3)+0.5*ones(3),1:3); F(2)

function F = sub_fn_mi(sigma,V)
    F.sigma = sigma;
    F.V = V;

    F.indsA = [];
    F.invAc = [];
    F.indsAc = [];
    F.cholA = [];

    F = class(F,'sub_fn_mi',sfo_fn);
    F = set(F,'current_set',-1);
end