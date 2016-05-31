clear; clc; close all;
addpath('utility');
addpath('ElasticNet');
addpath('ElasticNet/LapSSEN2');
addpath('ElasticNet/imm');

%% parameters
option.dataset_id = 5;
option.kernel = 'ST'; 	% regularization: S: spatial, T: temporal, ST:temporal and spatial, N: none, A: all
option.what = 'semi-supervised';
% option.what = 'submodualr';
% option.term = 'mix'; 	% submodular method: random, facility, temporal, diversity, mix
% option.ratio = [0.05 0.1 0.15]; 	% 0.01 0.05 0.1 0.15 0.2

if option.dataset_id == 3 % mall dataset
	if strcmp(option.what, 'semi-supervised')==1
		option.LabelNum = [10 20 40 60 80];
		option.UnlabelNum = [0 100 200 400 600];
		% option.LabelNum = [10 20 30 40 50 60 70 80 90 100];
		% option.UnlabelNum = [400];
	else 	% for submodualr test
		option.TrainNum = 600;
		option.SelectNum = [10 20 30 40 50 60 70 80 90 100];
		option.bound = max(option.SelectNum);
		% option.Group = option.TrainNum*option.ratio;
		option.Group = [30 50 70];
		option.init_S = [];
	end
elseif option.dataset_id == 4 % ucsd dataset
	if strcmp(option.what, 'semi-supervised')==1
		option.LabelNum = [10 20 40 60 80];
		option.UnlabelNum = [0 100 200 400 600];
		% option.LabelNum = [10 20 30 40 50 60 70 80 90 100];
		% option.UnlabelNum = [400];
	else
		option.TrainNum = 700;
		option.SelectNum = [10 20 30 40 50 60 70 80 90 100];
		option.bound = max(option.SelectNum);
		% option.Group = option.TrainNum*option.ratio;
		option.Group = [30];
		% option.Group = [30 50];
		option.init_S = [];
	end
elseif option.dataset_id == 5 % fudan dataset
	if strcmp(option.what, 'semi-supervised')==1
		option.LabelNum = [10 20 40 60 80];
		option.UnlabelNum = [0 50 100 200 400];
		% option.LabelNum = [10 20 30 40 50 60 70 80 90 100];
		% option.UnlabelNum = [400];
	else
		option.TrainNum = 500;
		option.SelectNum = [10 20 30 40 50 60 70 80 90 100];
		option.bound = max(option.SelectNum);
		% option.Group = option.TrainNum*option.ratio;
		option.Group = [10 20 30];
		option.init_S = [];
	end
end

option.lambdaRegSet = 10.^(-2:0.3:1);
option.widthSset = 10.^(-2:0.2:0); 		% tune for spatial kernel and lambda_S
option.widthTset = 10.^(-2:0.2:0); 		% tune for temporal kernel and lambda_T
% option.alphaSet = 10.^(-2:0.2:0);		% tune for elastic net
option.thetaSet = [0.2 0.4 0.6 0.8];	% proportion of temporal kernel
option.theta = 0;

option.nRepeat = 20;
option.normalize = 1;
option.tuneMethod = 1;  % 1: cross validation 2: leave one out 3: use true label
option.nfold = 5;
option.KNN = 10;	% neighbor of spatial
option.KTT = 3; 	% neighbor of temporal

%% load dataset
[trainFrame, testFrame, dataset_feature, dataset_ground_truth] = getDataset(option.dataset_id);
feaIndex = getFeatureIndex('all', option.dataset_id);
load(dataset_feature);
load(dataset_ground_truth);

Train.Feature = Feature(trainFrame,feaIndex);
Train.Truth = count(trainFrame);
Train.Time = [1:length(Train.Truth)]';
Test.Feature = Feature(testFrame,feaIndex);
Test.Truth = count(testFrame);
nTrain = size(Train.Feature,1);
nTest = size(Test.Feature,1);

% normalize X
if option.normalize == 1
    Xmean = mean(Train.Feature,1); Xstd = std(Train.Feature,1);
    Train.Feature = (Train.Feature - repmat(Xmean,nTrain,1))./repmat(Xstd,nTrain,1);
    Test.Feature = (Test.Feature - repmat(Xmean,nTest,1))./repmat(Xstd,nTest,1);
    Tmean = mean(Train.Time); Tstd = std(Train.Time);
    Train.Time = (Train.Time - repmat(Tmean,nTrain,1))./repmat(Tstd,nTrain,1);
end

% spatial similarity matrix
% option.K = pdist2(Train.Feature, Train.Feature).^2;
% option.T = pdist2(Train.Time, Train.Time).^2;
option.K = Inf(nTrain);
K = pdist2(Train.Feature, Train.Feature).^2;
[K, id] = sort(K, 2);
for i=1:nTrain
	option.K(i, id(i,1:option.KNN)) = K(i, id(i,1:option.KNN));
end

% temporal similarity matrix
option.T = Inf(nTrain);
T = pdist2(Train.Time, Train.Time).^2;
bound_s = [1:nTrain] - option.KTT; bound_s = max(bound_s, 1);
bound_l = [1:nTrain] + option.KTT; bound_l = min(bound_l, nTrain);
for i=1:nTrain
	option.T(i,bound_s(i):bound_l(i)) = T(i,bound_s(i):bound_l(i));
end
if option.dataset_id == 5 % fudan dataset
	T = Inf(nTrain);
	T(1:100,1:100) = option.T(1:100,1:100);
	T(101:200,101:200) = option.T(101:200,101:200);
	T(201:300,201:300) = option.T(201:300,201:300);
	T(301:400,301:400) = option.T(301:400,301:400);
	T(401:500,401:500) = option.T(401:500,401:500);
	option.T = T;
end

option

% Laplacian semi-supervised elastic net
warning off;
tic

if strcmp(option.what, 'semi-supervised')==1
	[result] = Wrap_rand_LapSSEN(Train, Test, option);
	% [result] = Wrap_rand_LapSSEN5(Train, Test, option);
elseif strcmp(option.what, 'submodualr')==1
	for iRep = 1:option.nRepeat
		index = randperm(nTrain);
		TrainIndex = sort(index(1:option.TrainNum));
		TrainFeature = Train.Feature(TrainIndex, :);
        % random
        idx = randperm(option.TrainNum);
        selectedIndex = TrainIndex(idx(1:option.bound));
        mse_array = Wrap_subm_LapSSEN(Train, Test, TrainIndex, selectedIndex, option);
        Result.random(iRep, :) = mse_array(:);

        % submodular
        Result.random = zeros(option.nRepeat, length(option.SelectNum));
        Result.submodualr = cell(length(option.theta), length(option.Group));
        for i=1:length(option.theta)
        	for j=1:length(option.Group)
        		Result.submodualr{i, j} = zeros(option.nRepeat, length(option.SelectNum));
        	end
        end
        sigmaK = exp(-option.K(TrainIndex, TrainIndex));
        sigmaT = exp(-option.T(TrainIndex, TrainIndex));
        for iTheta = 1:length(option.thetaSet)
        	option.theta = option.thetaSet(iTheta);
        	Sigma = (1-option.theta)*sigmaK + option.theta*sigmaT;
        	for iGroup = 1:length(option.Group)
        		Group = option.Group(iGroup);
        		cluster = SClustering(Sigma, Group);
        		% cluster = kmeans(TrainFeature, Group);
        		idx = GreedyMix(Sigma, cluster', Group, option.bound, []);
        		selectedIndex = TrainIndex(idx)
        		mse_array = Wrap_subm_LapSSEN(Train, Test, TrainIndex, selectedIndex, option);
        		Result.submodualr{iTheta, iGroup}(iRep, :) = mse_array(:);
        	end
        end
	end
	% [result] = Wrap_subm_LapSSEN(Train, Test, option);
	% [result] = Wrap_LapSSEN3(Train, Test, option);
end

toc

% save result_LapSSEN_ucsd result option