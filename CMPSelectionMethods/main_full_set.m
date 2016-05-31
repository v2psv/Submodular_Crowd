clear; clc; close all;
addpath('regression/imm'); addpath('regression/gp'); addpath('utility');
addpath('selection'); addpath('selection/ZPclustering');

%% generall parameters
normalize = 1;
nRepeat = 1;
dataset = 'ucsd';
regress = 'LapEN';

option.nfold = 10;
option.LabelNum = 800;

%  parameters for LapEN
option.KNN = 10;
option.kernel = 'ST';
option.lambdaASet = 10.^(-2:0.2:2);
option.lambdaISet = 10.^(-2:0.2:2);
option.thetaSet = [0:0.1:1];    % proportion of temporal kernel

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
  Labeled = Train;
  Ymean = mean(Labeled.Label);
  Labeled.Label = Labeled.Label - Ymean;
  SemiFeature = Train.Feature;
  SemiIdx = [1:nTrain];

  [mse, mae] = CV_LapEN(option.kernel, Labeled, SemiFeature, SemiIdx, Ymean, Test, option)
  MSE = [MSE; mse];
  MAE = [MAE; mae];
  MSE_MEAN = mean(MSE)
  MAE_MEAN = mean(MAE)
end

toc
