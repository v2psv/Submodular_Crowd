function [Train, Test, NPool] = loadDataset(dataset)
  if strcmp(dataset, 'ucsd')==1
    trainIndex = [601:1400];
    testIndex = [1:600 1401:2000];
    dataset_feature = 'dataset/cvpr_feat.mat';
    dataset_label = 'dataset/cvpr_gt.mat';
    NPool = 700;
  elseif strcmp(dataset, 'fudan')==1
    trainIndex = [1:100 301:400 601:700 901:1000 1201:1300];
    testIndex = [101:300 401:600 701:900 1001:1200 1301:1500];
    dataset_feature = 'dataset/fudan_feat.mat';
    dataset_label = 'dataset/fudan_gt.mat';
    NPool = 400;
  elseif strcmp(dataset, 'mall')==1
    trainIndex = [1:800];
    testIndex = [801:2000];
    dataset_feature = 'dataset/mall_feat2.mat';
    dataset_label = 'dataset/mall_gt.mat';
    NPool = 700;
  end

  load(dataset_feature); % Feature
  load(dataset_label);   % Label

  Train.Feature = Feature(trainIndex,:);
  Train.Label = Label(trainIndex);
  Train.Time = [1:length(Train.Label)]';
  Test.Feature = Feature(testIndex,:);
  Test.Label = Label(testIndex);
end
