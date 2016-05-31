function [params] = SSLasso3(TrainFeature, labelSet, unlabelSet, LabelTruth, opt)
    if nargin < 4
        lambda2Set = [0:0.1:1];
        lambda3Set = [0:0.1:1];
    else
        lambda2Set = opt.lambda2Set;
        lambda3Set = opt.lambda3Set;
    end

    nLabel = length(labelSet);
    nUnlabel = length(unlabelSet);
    LabelFeature = TrainFeature(labelSet, :);
    UnlabelFeature = TrainFeature(unlabelSet, :);
    SemiFeature = TrainFeature(1:end-1, :) - TrainFeature(2:end, :);
    ind = ones(1, size(TrainFeature,1));
    ind(labelSet) = 0; ind(unlabelSet) = 0;
    ind(1) = 0; ind(size(TrainFeature,1)-1) = 0;
    SemiFeature(ind,:) = []; SemiFeature(ind-1,:) = [];

    mse_array = zeros(length(lambda2Set), length(lambda3Set));

    for i=1:length(lambda2Set)
        for j=1:length(lambda3Set)
            lambda2 = lambda2Set(i);
            lambda3 = lambda3Set(j);

            % n fold cross validation
            tmp_err = 0;
            for k=1:nLabel/5
                ind = zeros(1, nLabel); ind((k-1)*5+1:k*5) = 1;
                itrain = ~ind; itest = ~itrain;

                X = [LabelFeature(itrain,:); sqrt(lambda3)*SemiFeature];
                Y = [LabelTruth(itrain); zeros(size(SemiFeature, 1), 1)];
                all_beta = larsen(X, Y, lambda2, 0, 0);

                predict = LabelFeature(itest,:) * all_beta';
                res = predict-repmat(LabelTruth(itest,:),1,size(predict,2));
                [min_err, ~] = min(mean(res.^2,1));
                tmp_err = tmp_err + min_err/nLabel;

                % predict = LabelFeature * all_beta';
                % res = predict-repmat(LabelTruth,1,size(predict,2));
                % [~, idx] = min(mean(res.^2, 1));
                % beta = all_beta(idx,:);
                % predict = LabelFeature(itest,:) * beta';
                % tmp_err = tmp_err + (predict-LabelTruth(itest,:))^2/nLabel;
            end
            mse_array(i,j) = tmp_err;
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