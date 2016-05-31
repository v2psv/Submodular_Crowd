clear; clc; close all;
addpath('gp');
addpath('utility');

%% parameters
dataset_id = 4;
covfunc_id = 'l';
gptrials = 1; gpnorm   = 'Xy';
TrainNum = [25];
dataset_subset = 'subset_index/subset_cvpr_facility_div5.mat';
features = ['all'];
% Area Perimeter        S
% Perimeter orientation	P
% Ratio                 R
% Edge                  E
% Edge orientation      Y
% Minkowski dimension	M
% GLCM                  G

%% load dataset
[trainFrame, testFrame, dataset_feature, dataset_ground_truth] = getDataset(dataset_id);
load(dataset_feature);
load(dataset_ground_truth);
load(dataset_subset);

load('JELSR_20160926_1102.mat');
Feature = Feature*RES(3,1,3,2,2).W;

if strcmp(covfunc_id,'l')==1
    covfunc = {'covLINone'};
elseif strcmp(covfunc_id,'r')==1
    covfunc = {'covSEiso'};
elseif strcmp(covfunc_id,'lr')==1
    covfunc = {'covSum', {'covLINone', 'covSEiso'}};
end

%% evaluation
mae_array = zeros(size(features,1), length(TrainNum));
mse_array = zeros(size(features,1), length(TrainNum));
% run('startup.m')
for iTrial = 1:size(features,1)
    % feat = features(iTrial,:);
    % feaIndex = getFeatureIndex(feat, dataset_id);
    feaIndex = [1:size(Feature,2)];
    % get evaluation set
    Xtrain = Feature(trainFrame,feaIndex);
    Xtest  = Feature(testFrame,feaIndex);
    Ytrain = count(trainFrame);
    Ytest  = count(testFrame);

    % GP regression
    for i=1:length(TrainNum)
        % trainIdx = S2(1:TrainNum(i));
        trainIdx = [1:800];
        X = Xtrain(trainIdx,:); Y = Ytrain(trainIdx);
        gpm = gp_train(X', Y', covfunc, gpnorm, gptrials);
        [Ypred_raw, Spred] = gp_predict(Xtest', gpm);
        Ypred = max(round(Ypred_raw), 0);   % truncate and round prediction

        mae_array(iTrial,i) = mean(abs(Ypred - Ytest(:)));
        mse_array(iTrial,i) = mean((Ypred - Ytest(:)).^2);
    end
end