function [mse, mae] = CV_SSEN(Labeled, SemiFeature, Ymean, Test, option);
    if nargin < 5
        error('In CV_SSEN:Too few parameters!');
    else
        nfold = min(option.nfold, size(Labeled.Feature, 1));
        lambda2Set = option.lambda2Set;
        lambda3Set = option.lambda3Set;
        nStep = 1000;
        step = 1/(nStep - 1);
    end

    % strip parameters for parallel computing
    strip_Lambda2 = [];
    strip_Lambda3 = [];
    for lambda2 = lambda2Set
        for lambda3 = lambda3Set
            strip_Lambda2 = [strip_Lambda2; lambda2];
            strip_Lambda3 = [strip_Lambda3; lambda3];
        end
    end

    [nLabel p] = size(Labeled.Feature);
    mse_array = zeros(length(strip_Lambda2),1);
    beta_array = zeros(length(strip_Lambda2), p);
    s_array = zeros(length(strip_Lambda2),1);

    indices = crossvalind('Kfold', nLabel, nfold);
    parfor i=1:length(strip_Lambda2)
        warning off;
        lambda2 = strip_Lambda2(i);
        lambda3 = strip_Lambda3(i);

        % cross validation
        res = zeros(nfold, nStep);
        cv_mse = zeros(nfold, 1);
        for k = 1:nfold
            itest = (indices==k); itrain = ~itest;
            X = [Labeled.Feature(itrain,:); sqrt(lambda3)*SemiFeature];
            Y = [Labeled.Label(itrain); zeros(size(SemiFeature, 1), 1)];

            beta = larsen(X, Y, lambda2, 0, 0);
            t = sum(abs(beta),2);
            s = (t - min(t))/max(t - min(t));
            [sm s_idx] = unique(s, 'rows');
            beta_interp = interp1q(s(s_idx), beta(s_idx, :), (0:step:1)');
            res(k, :) = sum((Labeled.Label(itest)*ones(1,nStep) - Labeled.Feature(itest,:)*beta_interp').^2);
        end
        res_mean = mean(res); res_std = std(res);
        [res_min idx_opt] = min(res_mean);

        %% Find optimal coefficient vector
        s_opt = idx_opt/nStep;
        beta = larsen(X, Y, lambda2, 0, 0);
        t = sum(abs(beta),2);
        s = (t - min(t))/max(t - min(t));
        [sm s_idx] = unique(s, 'rows');
        b_opt = interp1q(s(s_idx), beta(s_idx, :), s_opt);

        mse_array(i) = res_min;
        s_array(i) = s_opt;
        beta_array(i,:) = b_opt;
    end
    [~, id] = min(mse_array);
    opt_s = s_array(id);
    opt_lambda2 = strip_Lambda2(id);
    opt_lambda3 = strip_Lambda3(id);
    opt_beta = beta_array(id,:);

    predict = Test.Feature * opt_beta' + Ymean;
    predict = max(round(predict), 0);
    res = predict - Test.Label;

    mse = mean(res.^2);
    mae = mean(abs(res));
end
