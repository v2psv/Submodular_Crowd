clear; clc; close all;
addpath('utility');
addpath('ElasticNet');
addpath('ElasticNet/LapSSEN');
addpath('ElasticNet/imm');

%% parameters
option.normalize = 1;
option.tuneMethod = 1;  % 1: cross validation 2: leave one out 3: use true label
option.dataset_id = 3;  % 3: mall, 4: ucsd, 5: fudan
option.TrainNum = 600;
option.LabelNum = [10:10:100];
option.nfold = 10;
option.KNN = 5;
option.init_S = [];
option.bound = 100;
% option.groupNum = 40;
option.nRepeat = 24;
option.lambdaRegSet = [0 0.01 0.02 0.04 0.08 0.1 0.2 0.4 0.8 1 2 4 8 10 20 40 80 100];
option.widthSet = [0.001:0.001:0.01 0.01:0.01:0.1 0.1:0.1:1];
option.term = 'random';

%% load dataset
[trainFrame, testFrame, dataset_feature, dataset_ground_truth] = getDataset(option.dataset_id);
feaIndex = getFeatureIndex('all', option.dataset_id);
load(dataset_feature);
load(dataset_ground_truth);
% load(option.dataset_subset);

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

% Laplacian semi-supervised elastic net
warning off;
option.K = pdist2(Train.Feature, Train.Feature).^2;
option
tic


id = 1;
tic
% if strcmp(opt.submodular_func, 'facility_diversity')==1
%     for iLambda = 1:length(opt.lambdaSet)
%         lambda = opt.lambdaSet(iLambda);
%         F = sub_fn_facility_div(sigma, Ntrain, lambda, cluster);
%         [subset,scores] = greedy(F, Ntrain, opt.bound, opt.init_S);
%         DataIndex(id, :) = subset(:);
%         Scores(id, :) = scores(:);
%         params(id).group = Ngroup;
%         params(id).width = width;
%         params(id).lambda = lambda;
%         % params(id).cluster = cluster;
%         % params(id).similarity = sigma;
%         id = id + 1
%     end
% elseif strcmp(opt.submodular_func, 'diversity')==1
%     F = sub_fn_div(sigma,Ntrain,cluster);
%     [subset,scores] = greedy(F, Ntrain, opt.bound, opt.init_S);
%     DataIndex(id, :) = subset(:);
%     Scores(id, :) = scores(:);
%     params(id).group = Ngroup;
%     params(id).width = width;
%     id = id + 1
% elseif strcmp(opt.submodular_func, 'facility')==1
%     F = sub_fn_facility(sigma, Ntrain);
%     [subset,scores] = greedy(F, Ntrain, opt.bound, opt.init_S);
%     DataIndex(id, :) = subset(:);
%     Scores(id, :) = scores(:);
%     params(id).group = Ngroup;
%     params(id).width = width;
%     id = id + 1
% elseif strcmp(opt.submodular_func, 'temporal')==1
%     T = Ntrain';
%     sigma = pdist2(T, T).^2;
%     F = sub_fn_facility(sigma, Ntrain);
%     [subset,scores] = greedy(F, Ntrain, opt.bound, opt.init_S);
%     DataIndex(id, :) = subset(:);
%     Scores(id, :) = scores(:);
%     params(id).group = Ngroup;
%     params(id).width = width;
%     id = id + 1
% end

% option.cluster = kmeans(Train.Feature, option.groupNum);
% sigma = exp(-K/option.width);  % similarity matrix
% [res] = Wrap_LapSSEN(DataIndex, Train, Test, sigma, option);

[result] = Wrap_LapSSEN2(Train, Test, option);
toc

% save result_LapEN_ucsd4 RESULT option
