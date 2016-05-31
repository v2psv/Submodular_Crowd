clear; clc; close all;
addpath('utility');
addpath('ElasticNet');
addpath('ElasticNet/EN');
addpath('ElasticNet/SSEN');
addpath('ElasticNet/LapEN');
addpath('ElasticNet/imm');

%% parameters
dataset_id = 4;
TrainNum = [10:5:50];
nRepeat = 10;
dataset_subset = 'subset_index/subset_ucsd_FacDiv.mat';
feature = 'all';

%% load dataset
[trainFrame, testFrame, dataset_feature, dataset_ground_truth] = getDataset(dataset_id);
feaIndex = getFeatureIndex(feature, dataset_id);
load(dataset_feature);
load(dataset_ground_truth);
load(dataset_subset);

Train.Feature = Feature(trainFrame,feaIndex);
Train.Truth = count(trainFrame);
Test.Feature = Feature(testFrame,feaIndex);
Test.Truth = count(testFrame);
nTrain = size(Train.Feature,1);
nTest = size(Test.Feature,1);

%% evaluation
% elastic net
% warning off;
% for iTrain=1:length(TrainNum)
%     for iIndex = 1:size(DataIndex,1)
%         labelSet = DataIndex(iIndex,1:TrainNum(iTrain));
%         Label.Feature = Train.Feature(labelSet,:);
%         Label.Truth = Train.Truth(labelSet);

%         % normalization
%         Xmean = mean(Train.Feature,1); Xstd = std(Train.Feature,1);
%         Ymean = mean(Label.Truth);
%         TrainFeature = (Train.Feature - repmat(Xmean,nTrain,1))./repmat(Xstd,nTrain,1);
%         TestFeature = (Test.Feature - repmat(Xmean,nTest,1))./repmat(Xstd,nTest,1);
%         LabelTruth = Label.Truth - Ymean;

%         % Elastic Net
%         params = EN0(TrainFeature, labelSet, LabelTruth, TestFeature, Test.Truth, Ymean);
%         predict_raw = TestFeature * params.opt_beta' + Ymean;
%         predict = max(round(predict_raw),0);
%         mae_en(iTrain, iIndex) = mean(abs(predict - Test.Truth));
%         mse_en(iTrain, iIndex) = mean((predict - Test.Truth).^2);

%         % Elastic Net
%         % params = EN2(TrainFeature, labelSet, LabelTruth);
%         % predict_raw = TestFeature * params.opt_beta' + Ymean;
%         % predict = max(round(predict_raw),0);
%         % mae_en(iTrain) = mean(abs(predict - Test.Truth));
%         % mse_en(iTrain) = mean((predict - Test.Truth).^2);
%     end
% end

% semi supervised elastic net
% warning off;
% mae_ssen = zeros(length(TrainNum),size(DataIndex,1));
% mse_ssen = zeros(length(TrainNum),size(DataIndex,1));
% for iTrain=1:length(TrainNum)
%     for iIndex = 1:size(DataIndex,1)
%         labelSet = DataIndex(iIndex, 1:TrainNum(iTrain));
%         Label.Feature = Train.Feature(labelSet,:);
%         Label.Truth = Train.Truth(labelSet);

%         % normalization
%         Xmean = mean(Train.Feature,1); Xstd = std(Train.Feature,1);
%         Ymean = mean(Label.Truth);
%         TrainFeature = (Train.Feature - repmat(Xmean,nTrain,1))./repmat(Xstd,nTrain,1);
%         TestFeature = (Test.Feature - repmat(Xmean,nTest,1))./repmat(Xstd,nTest,1);
%         LabelTruth = Label.Truth - Ymean;

%         % Semi-supervised Elastic Net

%         mae_table = zeros(nRepeat,1);
%         mse_table = zeros(nRepeat,1);
%         for iRep = 1:nRepeat
%             params = SSLasso2(TrainFeature, labelSet, LabelTruth);
%             predict_raw = TestFeature * params.opt_beta' + Ymean;
%             predict = max(round(predict_raw),0);
%             % mae_ssen(iTrain, iRep) = mean(abs(predict - Test.Truth));
%             % mse_ssen(iTrain, iRep) = mean((predict - Test.Truth).^2);
%           mae_table(iRep) = mean(abs(predict - Test.Truth));
%             mse_table(iRep) = mean((predict - Test.Truth).^2);
%             % [iTrain iRep mae_ssen(iTrain, iRep) mse_ssen(iTrain, iRep)]
%             [iTrain iRep mae_table(iRep) mse_table(iRep)]
%         end

%         params = SSLasso2(TrainFeature, labelSet, LabelTruth);
%         predict_raw = TestFeature * params.opt_beta' + Ymean;
%         predict = max(round(predict_raw),0);
%         mae_ssen(iTrain,iIndex) = mean(abs(predict - Test.Truth));
%         mse_ssen(iTrain),iIndex = mean((predict - Test.Truth).^2);
%         [mae_ssen(iTrain,iIndex) mse_ssen(iTrain,iIndex) params.lambda2 params.lambda3]
%     end
% end

% % semi-supervised elastic net (parallel)
% % matlabpool 9
% tic
% spmd
%     iTrain = labindex;
%     warning off;
%     mae_table = zeros(nRepeat,size(DataIndex,1));
%     mse_table = zeros(nRepeat,size(DataIndex,1));
%     % for iIndex = 1:size(DataIndex,1)
%     for iIndex = subIndex
%         [iTrain iIndex]
%         labelSet = DataIndex(iIndex, 1:TrainNum(iTrain));
%         Label.Feature = Train.Feature(labelSet,:);
%         Label.Truth = Train.Truth(labelSet);

%         % normalization
%         Xmean = mean(Train.Feature,1); Xstd = std(Train.Feature,1);
%         Ymean = mean(Label.Truth);
%         TrainFeature = (Train.Feature - repmat(Xmean,nTrain,1))./repmat(Xstd,nTrain,1);
%         TestFeature = (Test.Feature - repmat(Xmean,nTest,1))./repmat(Xstd,nTest,1);
%         LabelTruth = Label.Truth - Ymean;

%         for iRep = 1:nRepeat
%             params = SSLasso2(TrainFeature, labelSet, LabelTruth);
%             predict_raw = TestFeature * params.opt_beta' + Ymean;
%             predict = max(round(predict_raw),0);
%             mae_table(iRep,iIndex) = mean(abs(predict - Test.Truth));
%             mse_table(iRep,iIndex) = mean((predict - Test.Truth).^2);
%         end
%     end
%     mae_mean = mean(mae_table,1); mae_std = std(mae_table,1);
%     mse_mean = mean(mse_table,1); mse_std = std(mse_table,1);
% end

% MAE.mean = zeros(length(TrainNum),size(DataIndex,1));
% MSE.mean = zeros(length(TrainNum),size(DataIndex,1));
% MAE.std = zeros(length(TrainNum),size(DataIndex,1));
% MSE.std = zeros(length(TrainNum),size(DataIndex,1));
% for i=1:length(TrainNum)
%   MAE.mean(i,:) = mae_mean{i};
%   MSE.mean(i,:) = mse_mean{i};
%   MAE.std(i,:) = mae_std{i};
%   MSE.std(i,:) = mse_std{i};
% end
% toc

% Laplacian semi-supervised elastic net (parallel)
% matlabpool 9
tic
spmd
    iTrain = labindex;
    warning off;
    mae_table = zeros(nRepeat,size(DataIndex,1));
    mse_table = zeros(nRepeat,size(DataIndex,1));
    for iIndex = 1:size(DataIndex,1)
        [iTrain iIndex]
        labelSet = DataIndex(iIndex, 1:TrainNum(iTrain));
        Label.Feature = Train.Feature(labelSet,:);
        Label.Truth = Train.Truth(labelSet);

        % normalization
        Xmean = mean(Train.Feature,1); Xstd = std(Train.Feature,1);
        Ymean = mean(Label.Truth);
        TrainFeature = (Train.Feature - repmat(Xmean,nTrain,1))./repmat(Xstd,nTrain,1);
        TestFeature = (Test.Feature - repmat(Xmean,nTest,1))./repmat(Xstd,nTest,1);
        LabelTruth = Label.Truth - Ymean;

        % evaluation
        for iRep = 1:nRepeat
            para = LapEN1(TrainFeature, labelSet, LabelTruth, params(iIndex).similarity, params(iIndex).cluster);
            predict_raw = TestFeature * para.opt_beta' + Ymean;
            predict = max(round(predict_raw),0);
            mae_table(iRep,iIndex) = mean(abs(predict - Test.Truth));
            mse_table(iRep,iIndex) = mean((predict - Test.Truth).^2);
        end
    end
    mae_mean = mean(mae_table,1); mae_std = std(mae_table,1);
    mse_mean = mean(mse_table,1); mse_std = std(mse_table,1);
end

MAE.mean = zeros(length(TrainNum),size(DataIndex,1));
MSE.mean = zeros(length(TrainNum),size(DataIndex,1));
MAE.std = zeros(length(TrainNum),size(DataIndex,1));
MSE.std = zeros(length(TrainNum),size(DataIndex,1));
for i=1:length(TrainNum)
  MAE.mean(i,:) = mae_mean{i};
  MSE.mean(i,:) = mse_mean{i};
  MAE.std(i,:) = mae_std{i};
  MSE.std(i,:) = mse_std{i};
end
toc

save result_LapEN_ucsd