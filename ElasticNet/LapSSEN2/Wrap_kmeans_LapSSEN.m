function [mse_array] = Wrap_kmeans_LapSSEN(Train, Test, TrainIndex, option)
    if nargin < 4
        error('In Wrap_kmeans_LapSSEN:Too few parameters!');
    end

    mse_array = zeros(1, length(option.SelectNum));
    for i = 1:length(option.SelectNum)
        nLabel = option.SelectNum(i);
        cluster = kmeans(Train.Feature(TrainIndex,:), nLabel);
        LabelIdx = [];
        for j = 1:nLabel
            CIdx = find(cluster==j);
            idx = randperm(length(CIdx));
            LabelIdx = [LabelIdx CIdx(idx(1))];
        end
        LabelFeature = Train.Feature(LabelIdx, :);
        LabelTruth = Train.Truth(LabelIdx, :);
        Ymean = mean(LabelTruth);
        LabelTruth = LabelTruth - Ymean;

        SemiIdx = TrainIndex;
        SemiFeature = Train.Feature(SemiIdx,:);

        mse_array(i) = CV_LapSSEN7(option.kernel, LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
    end
end