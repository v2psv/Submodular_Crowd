% [f] = PerformRegression(Labels, B, Lambda)
%
% INPUT:
% 'B': (N times N) regularization matrix calculated using 'ConstructHessian.m'
%       such that the Hessian energy of f is approximated as <f,Bf>.
% 'Labels.LFlag' 
%       if Labels.LFlag(i)=1, 'i'-th node is a labeled point.
%       if Labels.LFlag(i)=0, 'i'-th node is not a labeled point.
% 'Labels.y' = real column vector of size N containing labels. For the locations,
%       where the corresponding values of 'Labels.LFlag' are not 1,
%       the values of 'Labels.y' do not affect the regression.
% OUTPUT: 
% 'f':  learned function on the N points 
% 
% performs regression by minimizing the squared loss (only on the labeled
% points) plus Hessian energy <f,Bf> weighted by Lambda.
% 
% optionally, [f] = PerformRegression(Labels, B, Lambda1, BS, Lambda2)
% performs regression by minimizing the squared (only on the labeled
% points) plus stabilized Hessian energy <f,(B+(Lambda2/Lambda1)*BS)f> weighted by Lambda1.
% 'BS': (N times N) matrix-- can be computed using ConstructHessian.m
% 
% The stabilizer 'BS' lead in all our experiments not to a significantly
% better performance. Since one has to do cross-validation over the
% additional parameter Lambda2 we do not recommend its usage. 

function [F, B, BS] = PerformRegression(Labels, B, Lambda1, BS, Lambda2)

[NumNodes,NumNodes] = size(B);

if nargin <= 3
    A = Lambda1*B;
else
    A = Lambda1*B+Lambda2*BS;
end

[NumNodes] = length(Labels.y);
c = zeros(NumNodes,1);
for u=1:NumNodes
    if Labels.LFlag(u) > 0
        c(u) = Labels.y(u);
        A(u,u) = A(u,u)+1;
    end
end

F = A\c;
% F = minres(A,c,1e-9,1000);
% ResError = norm(A*F-c,2);
% display(sprintf('Res error=%f',ResError));
