clear; clc; close all;
addpath('utility');
addpath('ElasticNet');
addpath('ElasticNet/LapSSEN');
addpath('ElasticNet/imm');

%% parameters
option.normalize = 1;
option.tuneMethod = 1;  % 1: cross validation 2: leave one out 3: use true label
option.nfold = 10;
option.KNN = 5;
option.dataset_id = 4;
option.LabelNum = [5 10 20 40 80];
option.UnlabelNum = [0 200 400 600 800];
% option.LabelNum = [10 20 30 40 50];
% option.UnlabelNum = [0 200 400 600 800];
% option.lambdaLapSet = [0:0.1:1];
% option.widthSet = [0 0.001 0.003 0.009 0.01 0.03 0.09 0.1 0.3 0.6 0.9 1 3 6 9 10 30 60 90 100 300 600 900];                    % kernel width best:30 10
% option.widthSet = [0.001:0.003:1];
option.lambdaRegSet = [0 0.01 0.02 0.04 0.08 0.1 0.2 0.4 0.8 1 2 4 8 10 20 40 80 100];
option.widthSet = [0.001:0.001:0.01 0.01:0.01:0.1 0.1:0.1:1];
option.nRepeat = 48;

%% load dataset
[trainFrame, testFrame, dataset_feature, dataset_ground_truth] = getDataset(option.dataset_id);
feaIndex = getFeatureIndex('all', option.dataset_id);
load(dataset_feature);
load(dataset_ground_truth);

Train.Feature = Feature(trainFrame,feaIndex);
Train.Truth = count(trainFrame);
Test.Feature = Feature(testFrame,feaIndex);
Test.Truth = count(testFrame);
nTrain = size(Train.Feature,1);
nTest = size(Test.Feature,1);

% normalize X
if option.normalize == 1
    Xmean = mean(Train.Feature,1); Xstd = std(Train.Feature,1);
    Train.Feature = (Train.Feature - repmat(Xmean,nTrain,1))./repmat(Xstd,nTrain,1);
    Test.Feature = (Test.Feature - repmat(Xmean,nTest,1))./repmat(Xstd,nTest,1);
end

warning off;
option.K = pdist2(Train.Feature, Train.Feature).^2;
option
tic

% Laplacian semi-supervised elastic net
% option.cluster = kmeans(Train.Feature, option.groupNum);

% for iWidth = 1:length(option.widthSet)
%     width = option.widthSet(iWidth)
%     for iLambdaLap = 1:length(option.lambdaLapSet)
%         lambdaLap = option.lambdaLapSet(iLambdaLap);
%         % sigma = lambdaLap * exp(-K/width);  % similarity matrix
%         sigma = exp(-K/width);  % similarity matrix

%         for iAlpha = 1:length(option.alphaSet)
%             option.alpha = option.alphaSet(iAlpha)

%             res = Wrap_rand_LapSSEN(Train, Test, sigma, option);

%             Result{iWidth,iLambdaLap}{iAlpha} = res;
%         end
%     end
% end


% [result{iWidth} info{iWidth}] = Wrap_rand_LapSSEN2(Train, Test, sigma, option);
[result] = Wrap_rand_LapSSEN4(Train, Test, option);

toc

% save result_LapEN_ucsd4 RESULT option