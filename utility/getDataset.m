function [trainFrame, testFrame, dataset_feature, dataset_ground_truth] = getDataset(dataset_id)
    if dataset_id ==0
        trainFrame = [601:1400];
        testFrame = [1:600 1401:2000];
        dataset_feature = 'dataset/ucsd_f_feature_holistic.mat';
        dataset_ground_truth = 'dataset/ucsd_f_GroundTruth.mat';
    elseif dataset_id ==1
        trainFrame = [1401:2600];
        testFrame = [1:1400 2601:4000];
        % dataset_feature = 'dataset/ucsd_f_feature_holistic.mat';
        % dataset_ground_truth = 'dataset/ucsd_f_GroundTruth.mat';
        dataset_feature = 'dataset/Pedes1_feat.mat';
        dataset_ground_truth = 'dataset/Pedes1_gt.mat';
    elseif dataset_id ==2
        trainFrame = [1501:2500];
        testFrame = [1:1500 2501:4000];
        % dataset_feature = 'dataset/ucsd_d_feature_holistic.mat';
        % dataset_ground_truth = 'dataset/ucsd_d_GroundTruth.mat';
        dataset_feature = 'dataset/Pedes2_feat.mat';
        dataset_ground_truth = 'dataset/Pedes2_gt.mat';
    elseif dataset_id ==3
        trainFrame = [1:800];
        testFrame = [801:2000];
        dataset_feature = 'dataset/mall_feat2.mat';
        dataset_ground_truth = 'dataset/mall_gt.mat';
    elseif dataset_id ==4
        trainFrame = [601:1400];
        testFrame = [1:600 1401:2000];
        dataset_feature = 'dataset/cvpr_feat.mat';
        dataset_ground_truth = 'dataset/cvpr_gt.mat';
    elseif dataset_id ==5
        trainFrame = [1:100 301:400 601:700 901:1000 1201:1300];
        testFrame = [101:300 401:600 701:900 1001:1200 1301:1500];
        dataset_feature = 'dataset/fudan_feat.mat';
        dataset_ground_truth = 'dataset/fudan_gt.mat';
    end

end