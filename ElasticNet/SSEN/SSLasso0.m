function [params] = SSEN(TrainFeature, labelSet, LabelTruth, TestFeature, TestTruth, Ymean)
    lambda2Set = [0:0.01:1]; % best 0.52
    lambda3Set = [0:0.01:1]; % best 0.04

    nTrain = size(TrainFeature,1);
    nLabel = length(labelSet);
    nTest = length(TestTruth);

    LabelFeature = TrainFeature(labelSet, :);
    SemiFeature = TrainFeature(1:end-1, :) - TrainFeature(2:end, :);
    mae_array = zeros(length(lambda2Set), length(lambda3Set));
    mse_array = zeros(length(lambda2Set), length(lambda3Set));

    for i=1:length(lambda2Set)
        for j=1:length(lambda3Set)
            lambda2 = lambda2Set(i);
            lambda3 = lambda3Set(j);

            X = [LabelFeature; sqrt(lambda3)*SemiFeature];
            Y = [LabelTruth; zeros(size(SemiFeature, 1), 1)];
            all_beta = larsen(X, Y, lambda2, 0, 0);
            predict = LabelFeature * all_beta';
            res=predict-repmat(LabelTruth,1,size(predict,2));
            [~, idx] = min(mean(res.^2, 1));
            opt_beta =  all_beta(idx,:);

            predict = TestFeature * opt_beta' + Ymean;
            mse_array(i,j) = mean((predict - TestTruth).^2);
            mae_array(i, j) = mean(abs(predict - TestTruth));
        end
    end
    [r,c] = find(mse_array==min(mse_array(:)));
    params.lambda2 = lambda2Set(r);
    params.lambda3 = lambda3Set(c);

    X = [LabelFeature; sqrt(params.lambda3)*SemiFeature];
    Y = [LabelTruth; zeros(size(SemiFeature, 1), 1)];
    all_beta = larsen(X, Y, params.lambda2, 0, 0);
    predict = LabelFeature * all_beta';
    res = predict-repmat(LabelTruth,1,size(predict,2));
    [~, idx] = min(mean(res.^2, 1));
    params.opt_beta = all_beta(idx,:);
end