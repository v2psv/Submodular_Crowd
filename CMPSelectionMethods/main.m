clear; clc; close all;
addpath('regression/imm'); addpath('regression/gp'); addpath('utility');
addpath('selection'); addpath('selection/ZPclustering');
addpath('regression/gp/gpml-matlab-v3.2-2013-01-15')

%% generall parameters
normalize = 1;
nRepeat = 50;
dataset = 'fudan';
methods = {'random', 'k-means', 'm-landmark', 'submodular_mix'};
regress = 'SSEN';

option.nfold = 5;
option.LabelNum = 50;
%  parameters for SSEN
% option.lambda2Set = 10.^(-1:0.2:1); option.lambda3Set = 10.^(-1:0.2:1);
option.lambda2Set = [0:0.1:1]; option.lambda3Set = [0:0.1:1];
%  parameters for GPR
option.covfunc_id = 'l'; option.gptrials = 1; option.gpnorm   = 'Xy';
%  parameters for submodular
option.kernel = 'ST'; option.Group = 2; option.init_S = [];
option.theta = 0.5;	% proportion of temporal kernel
option.KNN = 5;	% neighbor of spatial

%% load dataset
% NPool: number of samples for selection pool
[Train, Test, NPool] = loadDataset(dataset);
nTrain = size(Train.Feature,1); nTest = size(Test.Feature,1);

% normalize X
if normalize == 1
    Xmean = mean(Train.Feature,1); Xstd = std(Train.Feature,1);
    Train.Feature = (Train.Feature - repmat(Xmean,nTrain,1))./repmat(Xstd,nTrain,1);
    Test.Feature = (Test.Feature - repmat(Xmean,nTest,1))./repmat(Xstd,nTest,1);
    Tmean = mean(Train.Time); Tstd = std(Train.Time);
    Train.Time = (Train.Time - repmat(Tmean,nTrain,1))./repmat(Tstd,nTrain,1);
end

option.K = pdist2(Train.Feature, Train.Feature).^2;
option.T = pdist2(Train.Time, Train.Time).^2;
option.sigmaK = exp(-option.K);
option.sigmaT = exp(-option.T);

tic
MSE = []; MAE = [];
for iRep = 1:nRepeat
  index = randperm(nTrain);
  option.PoolIndex = index(1:NPool);

  [mse_array, mae_array] = wrapSelection(Train, Test, methods, regress, option);
  MSE = [MSE; mse_array];
  MAE = [MAE; mae_array];
  MSE_MEAN = mean(MSE, 1)
  MAE_MEAN = mean(MAE, 1)
end

toc
