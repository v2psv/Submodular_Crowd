function [params] = EN1(TrainFeature, labelSet, LabelTruth, opt)
    if nargin < 4
        params.lambda2 = 0.2;
    else
        params.lambda2 = opt;
    end

    LabelFeature = TrainFeature(labelSet, :);

    X = [LabelFeature];
    Y = [LabelTruth];

    beta = larsen(X, Y, params.lambda2, 0, 0);
    predict_raw = LabelFeature * beta';
    predict = round(predict_raw);
    res=predict-repmat(LabelTruth,1,size(predict,2));
    [mse, idx] = min(mean(res.^2, 1));
    params.opt_beta = beta(idx,:);
end