clear; clc; close all;
addpath('utility');
addpath('ElasticNet');
addpath('ElasticNet/LapSSEN2');
addpath('ElasticNet/imm');

%% parameters
option.dataset_id = 4;
option.kernel = 'ST'; 	% regularization: S: spatial, T: temporal, ST:temporal and spatial, N: none, A: all
% option.what = 'semi-supervised';
option.what = 'submodular';
% option.what = 'submodular_term'; 	% submodular terms: random, facility, local_facility, diversity, mix
if option.dataset_id == 1 % Pedes1 dataset
	if strcmp(option.what, 'semi-supervised')==1
		option.LabelNum = [50];
		option.UnlabelNum = [1150];
		% option.LabelNum = [10 20 30 40 50 60 70 80 90 100];
		% option.UnlabelNum = [400];
	else
		option.TrainNum = 1000;
		% option.SelectNum = [10:5:50];
		option.SelectNum = [50];
		option.Unlabeled = 200;
		option.bound = max(option.SelectNum);
		option.Group = [2 5 10 15];
		option.init_S = [];
	end
elseif option.dataset_id == 2 % Pedes2 dataset
	if strcmp(option.what, 'semi-supervised')==1
		option.LabelNum = [50];
		option.UnlabelNum = [950];
		% option.LabelNum = [10 20 30 40 50 60 70 80 90 100];
		% option.UnlabelNum = [400];
	else
		option.TrainNum = 700;
		% option.SelectNum = [10:5:50];
		option.SelectNum = [50];
		option.Unlabeled = 200;
		option.bound = max(option.SelectNum);
		option.Group = [2 5 10 15];
		option.init_S = [];
	end
elseif option.dataset_id == 3 % mall dataset
	if strcmp(option.what, 'semi-supervised')==1
		option.LabelNum = [50];
		option.UnlabelNum = [750];
		% option.LabelNum = [10 20 30 40 50 60 70 80 90 100];
		% option.UnlabelNum = [400];
	else
		option.TrainNum = 700;
		% option.SelectNum = [10:5:50];
		option.SelectNum = [50];
		% option.Unlabeled = 200;
		option.bound = max(option.SelectNum);
		% option.Group = [2 5 10 15];
		option.Group = [5];
		option.init_S = [];
	end
elseif option.dataset_id == 4 % ucsd dataset
	if strcmp(option.what, 'semi-supervised')==1
		option.LabelNum = [50];
		option.UnlabelNum = [750];
		% option.LabelNum = [10 20 30 40 50 60 70 80 90 100];
		% option.UnlabelNum = [400];
	else
		option.TrainNum = 700;
		% option.SelectNum = [10 20 30 40 50 60 70 80 90 100];
		% option.SelectNum = [10:5:50];
		option.SelectNum = [50];
		option.bound = max(option.SelectNum);
		option.Group = [2 5 10 15];
		% option.Group = [5];
		% option.Unlabeled = 200;
		option.init_S = [];
	end
elseif option.dataset_id == 5 % fudan dataset
	if strcmp(option.what, 'semi-supervised')==1
		option.LabelNum = [50];
		option.UnlabelNum = [450];
		% option.LabelNum = [10 20 30 40 50 60 70 80 90 100];
		% option.UnlabelNum = [400];
	else
		option.TrainNum = 500;
		% option.SelectNum = [10 20 30 40 50 60 70 80 90 100];
		% option.SelectNum = [10:5:50];
		option.SelectNum = [50];
		% option.Unlabeled = 200;
		option.bound = max(option.SelectNum);
		option.Group = [2 5 10 15];
		% option.Group = [5];
		option.init_S = [];
	end
end

option.testTheta = [0 0.2 0.4 0.6 0.8 1];
% option.testTheta = 0.5;
option.lambdaASet = 10.^(-1:0.2:1);
option.lambdaISet = 10.^(-1:0.2:1);
option.thetaSet = [0:0.2:1];	% proportion of temporal kernel

option.nRepeat = 50;
option.normalize = 1;
option.tuneMethod = 1;  % 1: cross validation 2: leave one out 3: use true label
option.nfold = 10;
option.KNN = 5;	% neighbor of spatial
% option.KTT = 3; 	% neighbor of temporal

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


option.K = pdist2(Train.Feature, Train.Feature).^2;
option.T = pdist2(Train.Time, Train.Time).^2;
option.sigmaK = exp(-option.K);
option.sigmaT = exp(-option.T);

% spatial similarity matrix
% option.K = Inf(nTrain);
% K = pdist2(Train.Feature, Train.Feature).^2;
% [K, id] = sort(K, 2);
% for i=1:nTrain
% 	option.K(i, id(i,1:option.KNN)) = K(i, id(i,1:option.KNN));
% end

% temporal similarity matrix
% option.T = Inf(nTrain);
% T = pdist2(Train.Time, Train.Time).^2;
% bound_s = [1:nTrain] - option.KTT; bound_s = max(bound_s, 1);
% bound_l = [1:nTrain] + option.KTT; bound_l = min(bound_l, nTrain);
% for i=1:nTrain
% 	option.T(i,bound_s(i):bound_l(i)) = T(i,bound_s(i):bound_l(i));
% end
if option.dataset_id == 5 % fudan dataset have 5 sequences
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
elseif strcmp(option.what, 'submodular_term')==1
	Result.random = zeros(option.nRepeat, length(option.SelectNum));
	Result.facility = zeros(option.nRepeat, length(option.SelectNum));
	Result.locfac = zeros(option.nRepeat, length(option.SelectNum));
	Result.diversity = zeros(option.nRepeat, length(option.SelectNum));
	Result.mix = zeros(option.nRepeat, length(option.SelectNum));

	for iRep = 1:option.nRepeat
		iRep
		index = randperm(nTrain);
		TrainIndex = sort(index(1:option.TrainNum));

	    % random
	    idx = randperm(option.TrainNum);
	    selectedIndex = TrainIndex(idx(1:option.bound));
	    mse_array = Wrap_subm_LapSSEN(Train, Test, TrainIndex, selectedIndex, option);
	    Result.random(iRep, :) = mse_array(:);
	    MSE_random = mean(Result.random(1:iRep,:), 1);
	    disp(MSE_random);

		% facility
		theta = option.testTheta;
		Sigma = (1-theta)*option.sigmaK(TrainIndex,TrainIndex) + theta*option.sigmaT(TrainIndex,TrainIndex);
	    selectedIndex = GreedyFacility(Sigma, option.bound, []);
	    mse_array = Wrap_subm_LapSSEN(Train, Test, TrainIndex, selectedIndex, option);
	    Result.facility(iRep, :) = mse_array(:);
	    MSE_facility = mean(Result.facility(1:iRep,:), 1);
	    disp(MSE_facility);

	    %% submodular
	    theta = option.testTheta;
	    Group = option.Group(1);
	    Sigma = (1-theta)*option.sigmaK(TrainIndex,TrainIndex) + theta*option.sigmaT(TrainIndex,TrainIndex);
	    Cluster = SClustering(Sigma, Group);
	    % locfac
	    index = GreedyMix(Sigma, Cluster', Group, option.bound, [], 'locfac');
		selectedIndex = TrainIndex(index);
		mse_array = Wrap_subm_LapSSEN(Train, Test, TrainIndex, selectedIndex, option);
		Result.locfac(iRep, :) = mse_array(:);
		MSE_locfac = mean(Result.locfac(1:iRep,:), 1);
		disp(MSE_locfac);
		% diversity
	    index = GreedyMix(Sigma, Cluster', Group, option.bound, [], 'diversity');
		selectedIndex = TrainIndex(index);
		mse_array = Wrap_subm_LapSSEN(Train, Test, TrainIndex, selectedIndex, option);
		Result.diversity(iRep, :) = mse_array(:);
		MSE_diversity = mean(Result.diversity(1:iRep,:), 1);
		disp(MSE_diversity);
		% mix
	    index = GreedyMix(Sigma, Cluster', Group, option.bound, []);
		selectedIndex = TrainIndex(index);
		mse_array = Wrap_subm_LapSSEN(Train, Test, TrainIndex, selectedIndex, option);
		Result.mix(iRep, :) = mse_array(:);
		MSE_mix = mean(Result.mix(1:iRep,:), 1);
		disp(MSE_mix);
	end

	Result.mean_random = mean(Result.random, 1);
	Result.mean_facility = mean(Result.facility, 1);
	Result.mean_locfac = mean(Result.locfac, 1);
	Result.mean_diversity = mean(Result.diversity, 1);
	Result.mean_mix = mean(Result.mix, 1);

elseif strcmp(option.what, 'submodular')==1
	Result.random = zeros(option.nRepeat, length(option.SelectNum));
	Result.kmeans = zeros(option.nRepeat, length(option.SelectNum));
	Result.landmark = zeros(option.nRepeat, length(option.SelectNum));
	Result.submodular = cell(length(option.testTheta), length(option.Group));
	for i=1:length(option.testTheta)
		for j=1:length(option.Group)
			Result.submodular{i, j} = zeros(option.nRepeat, length(option.SelectNum));
		end
	end

	for iRep = 1:option.nRepeat
		iRep
		index = randperm(nTrain);
		TrainIndex = sort(index(1:option.TrainNum));

	    % random
	    % idx = randperm(option.TrainNum);
	    % selectedIndex = TrainIndex(idx(1:option.bound));
	    % mse_array = Wrap_subm_LapSSEN(Train, Test, TrainIndex, selectedIndex, option);
	    % Result.random(iRep, :) = mse_array(:);
	    % MSE_random = mean(Result.random(1:iRep,:), 1);
	    % disp(MSE_random);

	    % % k-means
	    % mse_array = Wrap_kmeans_LapSSEN(Train, Test, TrainIndex, option);
	    % Result.kmeans(iRep, :) = mse_array(:);
	    % MSE_kmeans = mean(Result.kmeans(1:iRep,:), 1);
	    % disp(MSE_kmeans);

	    % % landmark
	    % mse_array = Wrap_landmark_LapSSEN(Train, Test, TrainIndex, option);
	    % Result.landmark(iRep, :) = mse_array(:);
	    % MSE_landmark = mean(Result.landmark(1:iRep,:), 1);
	    % disp(MSE_landmark);

	    % submodular
	    for iTheta = 1:length(option.testTheta)
	    	theta = option.testTheta(iTheta);
	    	Sigma = (1-theta)*option.sigmaK(TrainIndex,TrainIndex) + theta*option.sigmaT(TrainIndex,TrainIndex);

	    	for iGroup = 1:length(option.Group)
	    		Group = option.Group(iGroup);
	    		Cluster = SClustering(Sigma, Group);
	    		index = GreedyMix(Sigma, Cluster', Group, option.bound, []);

	    		selectedIndex = TrainIndex(index);
        		mse_array = Wrap_subm_LapSSEN(Train, Test, TrainIndex, selectedIndex, option);
        		Result.submodular{iTheta, iGroup}(iRep, :) = mse_array(:);
        		% MSE_submodular = mean(Result.submodular{i, j}(1:iRep,:), 1);
        		% disp(MSE_submodular);
	    	end
	    end

	    % output information
	    MSE_submodular = zeros(length(option.testTheta), length(option.Group));
	    for i=1:length(option.testTheta)
	    	for j=1:length(option.Group)
	    		MSE_submodular(i, j) = mean(Result.submodular{i, j}(1:iRep,:), 1);
	    	end
	    end
	    disp(MSE_submodular);
	end
	Result.mean_random = mean(Result.random, 1);
	Result.mean_kmeans = mean(Result.kmeans, 1);
	Result.mean_landmark = mean(Result.landmark, 1);
    for i=1:length(option.testTheta)
    	for j=1:length(option.Group)
    		Result.mean_submodular{i, j} = mean(Result.submodular{i, j}, 1);
    	end
    end
end
toc