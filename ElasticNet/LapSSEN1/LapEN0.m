function [result] = LapEN0(Label, Test, Ymean, SemiFeature, option)
    if nargin < 5
        lambda2Set = [0:0.1:1];
        lambda3Set = [0:0.1:1];
        strip_lambda2Set = [];
        strip_lambda3Set = [];
        for lambda2 = lambda2Set
            for lambda3 = lambda3Set
                strip_lambda2Set = [strip_lambda2Set; lambda2];
                strip_lambda3Set = [strip_lambda3Set; lambda3];
            end
        end
    else
        strip_lambda2Set = option.strip_lambda2Set;
        strip_lambda3Set = option.strip_lambda3Set;
    end

    nLabel = length(Label.Truth);
    mse_array = zeros(length(strip_lambda2Set),1);
    mae_array = zeros(length(strip_lambda2Set),1);

    parfor i=1:length(strip_lambda2Set)
        warning off;
        lambda2 = strip_lambda2Set(i);
        lambda3 = strip_lambda3Set(i);

        X = [Label.Feature; sqrt(lambda3)*SemiFeature];
        Y = [Label.Truth; zeros(size(SemiFeature, 1), 1)];
        all_beta = larsen(X, Y, lambda2, 0, 0);

        % predict = Label.Feature * all_beta';
        % res = predict-repmat(Label.Truth,1,size(predict,2));
        % [~, idx] = min(mean(res.^2, 1));
        % opt_beta =  all_beta(idx,:);
        % predict = Test.Feature * opt_beta' + Ymean;
        % mse_array(i) = mean((predict - Test.Truth).^2);
        % mae_array(i) = mean(abs(predict - Test.Truth));

        predict = Test.Feature * all_beta' + Ymean;
        mse_array(i) = mean((predict - Test.Truth).^2);
        mae_array(i) = mean(abs(predict - Test.Truth));
    end
    result.mse_min = min(mse_array);
    result.mae_min = min(mae_array);
end