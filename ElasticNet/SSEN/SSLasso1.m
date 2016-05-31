function [params] = SSLasso1(TrainFeature, labelSet, LabelTruth, opt)
    if nargin < 4
        params.lambda2 = 0.5;
        params.lambda3 = 0.1;
    else
        params.lambda2 = opt.lambda2;
        params.lambda3 = opt.lambda3;
    end

    LabelFeature = TrainFeature(labelSet, :);
    SemiFeature = TrainFeature(1:end-1, :) - TrainFeature(2:end, :);

    X = [LabelFeature; sqrt(params.lambda3)*SemiFeature];
    Y = [LabelTruth; zeros(size(SemiFeature, 1), 1)];
    X = [LabelFeature];
    Y = [LabelTruth];
    all_beta = larsen(X, Y, params.lambda2, 0, 0);
    predict = LabelFeature * all_beta';
    res = predict-repmat(LabelTruth,1,size(predict,2));
    [~, idx] = min(mean(res.^2, 1));
    params.opt_beta = all_beta(idx,:);
end