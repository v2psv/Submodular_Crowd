clear; clc; close all;
addpath('../');
addpath('../../dataset');

%% parameters
k_array =[8 40 80];
dataset_id = 4;
features = 'all';
% Area Perimeter        S
% Perimeter orientation	P
% Ratio                 R
% Edge                  E
% Edge orientation      Y
% Minkowski dimension	M
% GLCM                  G

%% load dataset
[trainFrame, testFrame, dataset_feature, dataset_ground_truth] = getDataset(dataset_id);
feaIndex = getFeatureIndex(features, dataset_id);
load(dataset_feature);
load(dataset_ground_truth);

%% get evaluation set
Xtrain = Feature(trainFrame,feaIndex);
Ytrain = count(trainFrame);

y = [];
for i=1:length(k_array)
    y = [y; kmeans(Xtrain',k_array(i))];
end
y
save clusters y k_array