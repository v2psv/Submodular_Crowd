% function [result] = Wrap_subm_LapSSEN(Train, Test, option)
%     if nargin < 3
%         error('In Wrap_LapEN(Feature, Truth, option):Too few parameters!');
%     end
%     nSelectNum = length(option.SelectNum);

%     MSE = cell(1, option.nRepeat);
%     MAE = cell(1, option.nRepeat);

%     for iRep = 1:option.nRepeat
%         index = randperm(length(Train.Truth));
%         TrainIndex = sort(index(1:option.TrainNum));

%         if strcmp(option.term, 'facility')==1
%             Ntrain = [1:option.TrainNum];
%             sigma = option.AM(TrainIndex,TrainIndex);
%             F = sub_fn_facility(sigma, Ntrain);
%             [subsetIndex, scores] = greedy(F, Ntrain, option.bound, option.init_S);
%         elseif strcmp(option.term, 'diversity')==1
%             Ntrain = [1:option.TrainNum];
%             sigma = option.AM(TrainIndex,TrainIndex);
%             ZERO_DIAG = ~eye(option.TrainNum);
%             sigma = sigma.*ZERO_DIAG;
%             [cluster,~,~] = cluster_rotate(sigma, option.Ncluster, 0, 1); %ZelnikPerona Rotation clustering
%             F = sub_fn_diversity(sigma,Ntrain,cluster{1});
%             [subsetIndex, scores] = greedy(F, Ntrain, option.bound, option.init_S);
%         elseif strcmp(option.term, 'temporal')==1
%             Ntrain = [1:option.TrainNum];
%             T = TrainIndex';
%             sigma = pdist2(T, T).^2;
%             F = sub_fn_temporal(sigma, Ntrain);
%             [subsetIndex, scores] = greedy(F, Ntrain, option.bound, option.init_S);
%         elseif strcmp(option.term, 'random')==1
%             idx = randperm(length(Train.Truth));
%             subsetIndex = idx(1:option.bound);
%         elseif strcmp(option.term, 'mix')==1
%             V = Train.Feature(TrainIndex,:);
%             Ntrain = [1:option.TrainNum];
%             % sigma = option.AM(TrainIndex,TrainIndex);
%             ZERO_DIAG = ~eye(option.TrainNum);
%             sigma = sigma.*ZERO_DIAG;
%             cluster = kmeans(V, option.Group);
%             [subsetIndex, scores] = GreedyMix(V(:,[1 10]), sigma, cluster, option.Group, option.bound, []);
%             subsetIndex'
%         end

%         mse_array = zeros(nSelectNum,1);
%         mae_array = zeros(nSelectNum,1);

%         SemiIdx = TrainIndex;
%         SemiFeature = Train.Feature(SemiIdx,:);
%         for i = 1:nSelectNum
%             nLabel = option.SelectNum(i);
%             LabelIdx = index(subsetIndex(1:nLabel));
%             LabelFeature = Train.Feature(LabelIdx, :);
%             LabelTruth = Train.Truth(LabelIdx, :);
%             Ymean = mean(LabelTruth);
%             LabelTruth = LabelTruth - Ymean;

%             [mse_array(i) mae_array(i)] = CV_LapSSEN(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
%             [iRep nLabel mse_array(i) mae_array(i)]
%         end
%         MSE{iRep} = mse_array;
%         MAE{iRep} = mae_array;
%     end
%     result.mse = [];
%     result.mae = [];
%     result.mse_mean = [];
%     result.mae_mean = [];
%     result.mse = cell2mat(MSE);
%     result.mae = cell2mat(MAE);
%     result.mse_mean = mean(result.mse, 2);
%     result.mae_mean = mean(result.mae, 2);
% end

% function [result] = Wrap_subm_LapSSEN(Train, Test, option)
%     if nargin < 3
%         error('In Wrap_LapEN(Feature, Truth, option):Too few parameters!');
%     end
%     nSelectNum = length(option.SelectNum);

%     MSE = cell(1, option.nRepeat);
%     MAE = cell(1, option.nRepeat);

%     for iRep = 1:option.nRepeat
%         index = randperm(length(Train.Truth));
%         TrainIndex = sort(index(1:option.TrainNum));
%         Ntrain = [1:option.TrainNum];
%         TrainFeature = Train.Feature(TrainIndex,:);

%         if strcmp(option.term, 'facility')==1
%             F = sub_fn_facility(sigma, Ntrain);
%             [subsetIndex, scores] = greedy(F, Ntrain, option.bound, option.init_S);
%         elseif strcmp(option.term, 'diversity')==1
%             F = sub_fn_diversity(sigma,Ntrain,cluster{1});
%             [subsetIndex, scores] = greedy(F, Ntrain, option.bound, option.init_S);
%         elseif strcmp(option.term, 'random')==1
%             idx = randperm(length(Train.Truth));
%             subsetIndex = idx(1:option.bound);
%         elseif strcmp(option.term, 'mix')==1
%             sigmaK = exp(-option.K(TrainIndex, TrainIndex));
%             sigmaT = exp(-option.T(TrainIndex, TrainIndex));
%             theta = 0.5;
%             Group = 60;
%             Sigma = (1-theta)*sigmaK + theta*sigmaT;
%             % cluster = kmeans(TrainFeature, option.Group);
%             cluster = SClustering(Sigma, Group);
%             [subsetIndex, scores] = GreedyMix(Sigma, cluster', Group, option.bound, []);
%             subsetIndex'
%         end

%         mse_array = zeros(nSelectNum,1);
%         mae_array = zeros(nSelectNum,1);

%         SemiIdx = TrainIndex;
%         SemiFeature = Train.Feature(SemiIdx,:);
%         for i = 1:nSelectNum
%             nLabel = option.SelectNum(i);
%             LabelIdx = index(subsetIndex(1:nLabel));
%             LabelFeature = Train.Feature(LabelIdx, :);
%             LabelTruth = Train.Truth(LabelIdx, :);
%             Ymean = mean(LabelTruth);
%             LabelTruth = LabelTruth - Ymean;

%             [mse_array(i) mae_array(i)] = CV_LapSSEN6(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
%             [iRep nLabel mse_array(i) mae_array(i)]
%         end
%         MSE{iRep} = mse_array;
%         MAE{iRep} = mae_array;
%     end
%     result.mse = [];
%     result.mae = [];
%     result.mse_mean = [];
%     result.mae_mean = [];
%     result.mse = cell2mat(MSE);
%     result.mae = cell2mat(MAE);
%     result.mse_mean = mean(result.mse, 2);
%     result.mae_mean = mean(result.mae, 2);
% end

function [mse_array] = Wrap_subm_LapSSEN(Train, Test, TrainIndex, selectedIndex, option)
    if nargin < 5
        error('In Wrap_subm_LapSSEN:Too few parameters!');
    end

    mse_array = zeros(1, length(option.SelectNum));
    % temp = setdiff(TrainIndex, selectedIndex); idx = randperm(length(temp));
    % UnlabelIdx = temp(idx(1:option.Unlabeled));
    % SemiIdx = TrainIndex;
    % SemiFeature = Train.Feature(SemiIdx,:);
    for i = 1:length(option.SelectNum)
        nLabel = option.SelectNum(i);
        LabelIdx = selectedIndex(1:nLabel);
        LabelFeature = Train.Feature(LabelIdx, :);
        LabelTruth = Train.Truth(LabelIdx, :);
        Ymean = mean(LabelTruth);
        LabelTruth = LabelTruth - Ymean;

        % SemiIdx = sort([LabelIdx UnlabelIdx]);
        SemiIdx = TrainIndex;
        SemiFeature = Train.Feature(SemiIdx,:);
        mse_array(i) = CV_LapSSEN7(option.kernel, LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
    end
end