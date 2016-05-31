function [params] = EN2(TrainFeature, labelSet, LabelTruth, opt)
    if nargin < 4
        lambda2Set = [0:0.1:1]; % best 0.3
    else
        lambda2Set = opt.lambda2Set;
    end

    nLabel = length(labelSet);
    LabelFeature = TrainFeature(labelSet, :);
    mse_array = zeros(length(lambda2Set), 1);

    for i = 1:length(lambda2Set)
        lambda2 = lambda2Set(i);

        % leave one out (LOO)
        tmp_err = 0;
        for k=1:nLabel
            ind = zeros(1, nLabel); ind(k) = 1;
            itrain = ~ind; itest = ~itrain;

            X = LabelFeature(itrain,:);
            Y = LabelTruth(itrain);
            all_beta = larsen(X, Y, lambda2, 0, 0);
            predict = LabelFeature(itest,:) * all_beta';
            res = predict-repmat(LabelTruth(itest,:),1,size(predict,2));
            [min_err, ~] = min(res.^2);
            tmp_err = tmp_err + min_err/nLabel;
        end
        mse_array(i) = tmp_err;
    end

    [~, idx] = min(mse_array);
    params.lambda2 = lambda2Set(idx);

    beta = larsen(LabelFeature, LabelTruth, params.lambda2, 0, 0);
    predict = LabelFeature * beta';
    res = predict-repmat(LabelTruth,1,size(predict,2));
    [~, idx] = min(mean(res.^2, 1));
    params.opt_beta =  beta(idx,:);
end