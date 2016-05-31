clear; clc; close all;

%% parameters
dataset = 'mall';
TrainNum = 50;
nRandom = 100;
covfunc_id = 'l'; gptrials = 1; gpnorm   = 'Xy';

%% load dataset
[Train, Test, NPool] = loadDataset(dataset);
nTrain = size(Train.Feature,1); nTest = size(Test.Feature,1);

if strcmp(covfunc_id,'l')==1
    covfunc = {'covLINone'};
elseif strcmp(covfunc_id,'r')==1
    covfunc = {'covSEiso'};
elseif strcmp(covfunc_id,'lr')==1
    covfunc = {'covSum', {'covLINone', 'covSEiso'}};
end

%% evaluation
mae_array = zeros(1, nRandom);
mse_array = zeros(1, nRandom);
run('startup.m')

% GP regression
for iRand = 1:nRandom
    index = randperm(nTrain);
    trainIdx = index(1:TrainNum);
    X = Train.Feature(trainIdx,:); Y = Train.Label(trainIdx);
    gpm = gp_train(X', Y', covfunc, gpnorm, gptrials);
    [Ypred_raw, Spred] = gp_predict(Test.Feature', gpm);
    Ypred = max(round(Ypred_raw), 0);           % truncate and round prediction

    mae_array(iRand) = mean(abs(Ypred - Test.Label(:)));
    mse_array(iRand) = mean((Ypred - Test.Label(:)).^2);
end
