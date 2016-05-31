function [nXtrain, Ytrain] = uniqfeats(nXtrain, Ytrain, mode)
% uniqfeats - remove redundant features
%
%   [nXtrain, Ytrain] = uniqfeats(nXtrain, Ytrain)
%
% 
% remove points that are the same (this happens when the scene is completely empty)
if nargin<3
  mode = 1
end

originalsize = length(Ytrain);

switch(mode)
 case 1
  % NEW METHOD 
  
  % concatenate input and outputs
  tmpdata = [nXtrain; Ytrain];

  % get unique rows
  [tmp, tmpi, tmpj] = unique(tmpdata', 'rows');
  if (length(tmpi) < length(Ytrain))
    % use only unique (input,output) pairs
    nXtrain = tmp(:,1:end-1)';  % extract nXtrain
    Ytrain  = tmp(:,end)';      % extract Ytrain
    fprintf('cleaning data: found %d/%d unique points\n', length(Ytrain), originalsize);
  else
    
    % all unique
    fprintf('cleaning data: ok\n');
  end
  
 case 2
  % OLD METHOD 
  
  % get unique rows
  [tmp, tmpi, tmpj] = unique(nXtrain', 'rows');  
  if (length(tmpi) < length(Ytrain))
    Ytrain_orig = Ytrain;
    foo = Ytrain;
    foo(tmpi) = [];
    unique(foo)
    
    % use only unique points
    nXtrain = tmp';
    Ytrain  = Ytrain(tmpi);   % extract counts
    fprintf('cleaning data (old): found %d/%d unique points\n', length(Ytrain), originalsize);
  else
    
    % all unique
    fprintf('cleaning data: ok\n');
  end
end
 
 
