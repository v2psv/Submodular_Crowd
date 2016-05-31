function [params] = EN0(TrainFeature, labelSet, LabelTruth, TestFeature, TestTruth, Ymean)
    lambda2Set = [0:0.01:1];    % best 0.47

    LabelFeature = TrainFeature(labelSet, :);

    X = LabelFeature;
    Y = LabelTruth;

    mse_array = zeros(length(lambda2Set),1);
    for i = 1:length(lambda2Set)
        lambda2 = lambda2Set(i);
        beta = larsen(X, Y, lambda2, 0, 0);
        predict = LabelFeature * beta';
        res=predict-repmat(LabelTruth,1,size(predict,2));
        [~, idx] = min(mean(res.^2, 1));
        opt_beta =  beta(idx,:);

        predict = TestFeature * opt_beta' + Ymean;
        mse_array(i) = mean((predict - TestTruth).^2);
    end
    [min_mse, idx] = min(mse_array);
    params.lambda2 = lambda2Set(idx);
    beta = larsen(X, Y, params.lambda2, 0, 0);
    predict = LabelFeature * beta';
    res=predict-repmat(LabelTruth,1,size(predict,2));
    [~, idx] = min(mean(res.^2, 1));
    params.opt_beta =  beta(idx,:);
end