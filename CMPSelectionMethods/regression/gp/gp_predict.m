function [Ypred, Spred] = gp_predict(X, gpm)
% gp_predict - GPML wrapper function to predict with Gaussian Process regression
%
%   [Ypred, Spred] = gp_predict(X, gpm)
%
%       X  = feature vectors (each column is a feature vector)
%      gpm = Gaussian process learned with gp_train
%
%   Ypred = Y predictions
%   Spred = the variance of the predictions
%
% The function uses the GPML library (v3.2) by C. E. Rasmussen.  Make sure that GPML is 
% accessible in the path.
%
% Copyright 2008, Antoni Chan, SVCL, UCSD



% normalize input
if any(gpm.normmode == 'x') || any(gpm.normmode == 'X')
  X = X - repmat(gpm.Xmean,1,size(X,2));
  if any(gpm.normmode == 'X')
    X = X./repmat(gpm.Xstd,1,size(X,2));
  end  
end

% get predictions
[Ypred Spred] = gp(gpm.loghyper, @infExact, [], gpm.covfunc, @likGauss, gpm.X', gpm.Y(:), X');

% undo normalization
if any(gpm.normmode == 'y')
  Ypred = Ypred + gpm.Ymean;   % add mean to Y
end

