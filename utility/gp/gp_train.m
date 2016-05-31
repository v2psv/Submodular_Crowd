function gpm = gp_train(X, Y, covfunc, normmode, numtrials, initloghyper, uniqfy)
% gp_train - GPML wrapper function to train a Gaussian Process regression function
%
%    gpm = gp_train(X, Y, covfunc, normmode, numtrials, initloghyper, uniqfy)
% 
%         X = feature vectors (each column is a feature vector)
%         Y = function values
%   covfunc = covariance function ( default = {'covSEiso'} );
%  normmode = normalization mode:
%             'y' - normalize Y to zero-mean
%             'x' - normalize X to zero-mean
%             'X' - normalize X to zero-mean, unit variance
% numtrials = number of random initializations (the first is always fixed)
%           = (if negative, then all are random)
% initloghyper = initial log-hyperparameters (optional)
% uniqfy       = remove redundant training features (optional) [default=1 yes], 0=no
%
% The function uses the GPML library (v3.2) by C. E. Rasmussen.  See GPML for details
% on the covariance function.
%
% Copyright 2008, Antoni Chan, SVCL, UCSD

% 2013-03-01: updated for new GPML v3.2, added redundant feature removal
% 2010-10-22: added optional loghyper0 input


if (nargin < 3)
  covfunc = {'covSEiso'};
end
if (nargin < 4)
  normmode = '';
end
if (nargin < 5)
  numtrials = 1;
end
if (nargin<6)
  initloghyper = [];
end
if (nargin<7)
  uniqfy = 1;
end

% check for GPML
if ~exist('gp')
  error('Could not find GPML in path!');
end

MAXIT = 500;

% Data preprocessing: normalize inputs, remove redundant features, normalize output.
% This the order used in the original code (although probably the normalization should happen after feature removal)

% Normalize the inputs
if any(normmode=='x') || any(normmode == 'X')
  Xmean = mean(X,2);
  X = X - repmat(Xmean,1,size(X,2));
  gpm.Xmean = Xmean;
  
  if any(normmode == 'X')
    Xstd = std(X,0,2);
    X = X./repmat(Xstd,1,size(X,2));
    gpm.Xstd = Xstd;
  end  
end

% clean data (remove redundant feature vectors)
if (uniqfy)
  X_orig  = X;
  Y_orig  = Y;
  [X, Y] = uniqfeats(X, Y, uniqfy);
end

% normalize output
if any(normmode=='y')
  Ymean = mean(Y);   
  Y = Y - Ymean;    % subtract mean from Y
  gpm.Ymean    = Ymean;
end

% learn the GP
D = size(X,1);
nump = eval(feval(covfunc{:}));
numl = eval(likGauss());

loghyper0.cov = [zeros(nump,1)];
loghyper0.lik = [zeros(numl,1)];

for i=1:abs(numtrials)
  if (i==1) && (numtrials > 0)
    if ~isempty(initloghyper)
      loghyper0 = initloghyper;
    end
  else
    loghyper0.cov = randn(size(loghyper0.cov));
    loghyper0.lik = randn(size(loghyper0.lik));
  end
  
  fprintf('*** trial %d/%d ***\n', i, numtrials);
  fprintf('loghyper0 = ');
  fprintf('%g ', [loghyper0.cov; loghyper0.lik]);
  fprintf('\n');
  
  try
    [loghyper, fX] = minimize(loghyper0, @gp, -MAXIT, @infExact, [], covfunc, @likGauss, X', Y(:));
  catch
    fprintf('*** ERROR: %s\n', lasterr);

    if (abs(numtrials) == 1)
      error('can not continue!');
    end
    fprintf('continuing anyway...\n');
    loghyper = [];
    fX       = inf;
  end
  mymarg = -fX(end);
  
  fprintf('loghyper=')
  fprintf('%g ', [loghyper.cov; loghyper.lik]);
  fprintf('\n');
  fprintf('marglikelihood = %g\n', mymarg);
  
  all_loghyper0{i} = loghyper0;
  all_marglike(i)  = mymarg;
  all_loghyper{i}  = loghyper;
end

% pick the best
[best, besti] = max(all_marglike);
loghyper  = all_loghyper{besti};
loghyper0 = all_loghyper0{besti};

fprintf('*** best %d: ML=%g\n', besti, best);

gpm.loghyper  = loghyper;
gpm.covfunc   = covfunc;
gpm.X         = X;
gpm.Y         = Y;
gpm.normmode  = normmode;
gpm.loghyper0 = loghyper0;
gpm.mll       = best;

if (isempty(loghyper))
  error('no good runs!');
end
