clear; clc; close all;
addpath('utility');
addpath('ElasticNet');
addpath('ElasticNet/EN');
addpath('ElasticNet/SSEN');
addpath('ElasticNet/imm');

%% parameters
option.normalize = 1;

option.nfold = 5;
option.dataset_id = 3;
option.LabelNum = [50];
option.lambda2Set = [0:0.1:1];
option.lambda3Set = [0:0.1:1];
option.nRepeat = 50;
% option.method = 'random';
option.method = 'k-means';

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

tic
if strcmp(option.method, 'random')==1
	[result] = Wrap_rand_SSEN(Train, Test, option);
elseif strcmp(option.method, 'k-means')==1
	[result] = Wrap_Kmeans_SSEN(Train, Test, option);
end
toc