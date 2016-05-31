function [result] = Wrap_rand_SSEN(Train, Test, option)
    if nargin < 3
        error('In Wrap_rand_SSEN:Too few parameters!');
    else
        nRepeat = option.nRepeat;
        LabelNum = option.LabelNum;
    end

    result.mse_mean = zeros(length(LabelNum),1); result.mse_std = zeros(length(LabelNum),1);
    result.mae_mean = zeros(length(LabelNum),1); result.mae_std = zeros(length(LabelNum),1);

    SemiFeature = Train.Feature(1:end-1, :) - Train.Feature(2:end, :);

    for iLabel = 1:length(LabelNum)
        nLabel = LabelNum(iLabel);

        mse = zeros(nRepeat, 1);
        mae = zeros(nRepeat, 1);

        parfor iRep = 1:nRepeat
            index = randperm(length(Train.Truth));
            LabelIdx = index(1:nLabel);

            LabelFeature = Train.Feature(LabelIdx, :);
            LabelTruth = Train.Truth(LabelIdx, :);
            Ymean = mean(LabelTruth);
            LabelTruth = LabelTruth - Ymean;

            [mse(iRep), mae(iRep)] = CV_SSEN(LabelFeature, LabelTruth, SemiFeature, Ymean, Test, option);
            disp([mse(iRep), mae(iRep)]);
        end

        result.mse_mean(iLabel) = mean(mse); result.mse_std(iLabel) = std(mse);
        result.mae_mean(iLabel) = mean(mae); result.mae_std(iLabel) = std(mae);
    end
end