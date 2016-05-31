clear; clc; close all;

%% parameters
dataset_id = 3;
TrainNum = [50];
nRandom = 100;
covfunc_id = 'l'; gptrials = 1; gpnorm   = 'Xy';
feature = 'all';
% Area Perimeter        S
% Perimeter orientation	P
% Ratio                 R
% Edge                  E
% Edge orientation      Y
% Minkowski dimension	  M
% GLCM                  G

%% load dataset
[trainFrame, testFrame, dataset_feature, dataset_ground_truth] = getDataset(dataset_id);
load(dataset_feature);
load(dataset_ground_truth);

if strcmp(covfunc_id,'l')==1
    covfunc = {'covLINone'};
elseif strcmp(covfunc_id,'r')==1
    covfunc = {'covSEiso'};
elseif strcmp(covfunc_id,'lr')==1
    covfunc = {'covSum', {'covLINone', 'covSEiso'}};
end

%% evaluation
mae_array = zeros(length(TrainNum), nRandom);
mse_array = zeros(length(TrainNum), nRandom);
run('startup.m')

feaIndex = getFeatureIndex(feature, dataset_id);

% get evaluation set
Xtrain = Feature(trainFrame,feaIndex);
Xtest  = Feature(testFrame,feaIndex);
Ytrain = count(trainFrame);
Ytest  = count(testFrame);

% GP regression
for i=1:length(TrainNum)
    for iRand = 1:nRandom
        index = randperm(length(trainFrame));
        trainIdx = index(1:TrainNum(i));
        X = Xtrain(trainIdx,:); Y = Ytrain(trainIdx);
        gpm = gp_train(X', Y', covfunc, gpnorm, gptrials);
        [Ypred_raw, Spred] = gp_predict(Xtest', gpm);
        Ypred = max(round(Ypred_raw), 0);           % truncate and round prediction

        mae_array(i,iRand) = mean(abs(Ypred - Ytest(:)));
        mse_array(i,iRand) = mean((Ypred - Ytest(:)).^2);
    end
end
Result.mae_table = mae_array; Result.mse_table = mse_array;
Result.mae_avg = mean(mae_array, 2); Result.mse_avg = mean(mse_array, 2);
