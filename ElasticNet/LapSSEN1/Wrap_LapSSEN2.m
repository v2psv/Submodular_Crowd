function [result] = Wrap_LapSSEN2(Train, Test, option)
    if nargin < 3
        error('In Wrap_LapEN(Feature, Truth, option):Too few parameters!');
    end
    nSelectNum = length(option.SelectNum);

    MSE = cell(1, option.nRepeat);
    MAE = cell(1, option.nRepeat);

    for iRep = 1:option.nRepeat
        index = randperm(length(Train.Truth));
        TrainIndex = sort(index(1:option.TrainNum));

        if strcmp(option.term, 'facility')==1
            Ntrain = [1:option.TrainNum];
            sigma = option.AM(TrainIndex,TrainIndex);
            F = sub_fn_facility(sigma, Ntrain);
            [subsetIndex, scores] = greedy(F, Ntrain, option.bound, option.init_S);
        elseif strcmp(option.term, 'diversity')==1
            Ntrain = [1:option.TrainNum];
            sigma = option.AM(TrainIndex,TrainIndex);
            ZERO_DIAG = ~eye(option.TrainNum);
            sigma = sigma.*ZERO_DIAG;
            [cluster,~,~] = cluster_rotate(sigma, option.Ncluster, 0, 1); %ZelnikPerona Rotation clustering
            F = sub_fn_diversity(sigma,Ntrain,cluster{1});
            [subsetIndex, scores] = greedy(F, Ntrain, option.bound, option.init_S);
        elseif strcmp(option.term, 'temporal')==1
            Ntrain = [1:option.TrainNum];
            T = TrainIndex';
            sigma = pdist2(T, T).^2;
            F = sub_fn_temporal(sigma, Ntrain);
            [subsetIndex, scores] = greedy(F, Ntrain, option.bound, option.init_S);
        elseif strcmp(option.term, 'random')==1
            index = randperm(length(Train.Truth));
            subsetIndex = index(1:option.bound);
        end

        mse_array = zeros(nSelectNum,1);
        mae_array = zeros(nSelectNum,1);

        SemiIdx = TrainIndex;
        SemiFeature = Train.Feature(SemiIdx,:);
        for i = 1:nSelectNum
            nLabel = option.SelectNum(i);
            LabelIdx = index(subsetIndex(1:nLabel));
            LabelFeature = Train.Feature(LabelIdx, :);
            LabelTruth = Train.Truth(LabelIdx, :);
            Ymean = mean(LabelTruth);
            LabelTruth = LabelTruth - Ymean;

            [mse_array(i), mae_array(i)] = CV_LapSSEN2_2(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
        end
        MSE{iRep} = mse_array;
        MAE{iRep} = mae_array;
    end
    result.mse = [];
    result.mae = [];
    result.mse_mean = [];
    result.mae_mean = [];
    result.mse = cell2mat(MSE);
    result.mae = cell2mat(MAE);
    result.mse_mean = mean(result.mse, 2);
    result.mae_mean = mean(result.mae, 2);
end


function [mse, mae] = CV_LapSSEN2_1(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option)
    if nargin < 7
        error('In CV_LapSSEN:Too few parameters!');
    else
        nfold = min(option.nfold, size(LabelFeature, 1));
        lambdaRegSet = option.lambdaRegSet;
        widthSet = option.widthSet;
        knn = option.KNN;
    end

    nLabel = size(LabelFeature, 1);
    mse_array = zeros(length(lambdaRegSet)*length(widthSet),1);

    strip_LambdaReg = [];
    strip_Width = [];
    for lambdaReg = lambdaRegSet
        for width = widthSet
            strip_LambdaReg = [strip_LambdaReg; lambdaReg];
            strip_Width = [strip_Width; width];
        end
    end

    parfor i=1:length(strip_LambdaReg)
        warning off;
        lambdaReg = strip_LambdaReg(i);
        width = strip_Width(i);
        sigma = exp(-option.K(SemiIdx, SemiIdx)/width);
        sigma = sqrt(sigma);
        LapFeature = getLapFeature(SemiFeature, sigma, knn);

        % cross validation
        cv_mse = zeros(nfold, 1);
        indices = crossvalind('Kfold', nLabel, nfold);
        for k = 1:nfold
            itest = (indices==k); itrain = ~itest;

            X = [LabelFeature(itrain,:); LapFeature];
            Y = [LabelTruth(itrain); zeros(size(LapFeature, 1), 1)];
            beta = larsen(X, Y, lambdaReg, 0, 0);
            predict = LabelFeature(itest,:) * beta';
            res = predict - repmat(LabelTruth(itest),1,size(predict,2));
            [min_err, ~] = min(mean(res.^2,1));
            cv_mse(k) = min_err;
        end
        mse_array(i) = mean(cv_mse);
    end
    [~, id] = min(mse_array);
    opt_lambdaReg = strip_LambdaReg(id);
    opt_width = strip_Width(id);
    sigma = exp(-option.K(SemiIdx, SemiIdx)/opt_width);
    sigma = sqrt(sigma);
    LapFeature = getLapFeature(SemiFeature, sigma, knn);

    X = [LabelFeature; LapFeature];
    Y = [LabelTruth; zeros(size(LapFeature, 1), 1)];
    beta = larsen(X, Y, opt_lambdaReg, 0, 0);
    predict = Test.Feature * beta' + Ymean;
    res = predict - repmat(Test.Truth,1,size(predict,2));
    [~, idx] = min(mean(res.^2, 1));

    mse = mean(res(:,idx).^2, 1);
    mae = mean(abs(res(:,idx)), 1);
    [nLabel opt_lambdaReg opt_width mse mae]
end

function [mse, mae] = CV_LapSSEN2_2(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option)
    if nargin < 7
        error('In CV_LapSSEN:Too few parameters!');
    else
        nfold = min(option.nfold, size(LabelFeature, 1));
        lambdaRegSet = option.lambdaRegSet;
        alphaSet = option.alphaSet;
        widthSet = option.widthSet;
        knn = option.KNN;
    end

    strip_Alpha = [];
    strip_Width = [];
    for alpha = alphaSet
        for width = widthSet
            strip_Alpha = [strip_Alpha; alpha];
            strip_Width = [strip_Width; width];
        end
    end

    nLabel = size(LabelFeature, 1);
    mse_array = zeros(length(alphaSet)*length(widthSet),1);

    indices = crossvalind('Kfold', nLabel, nfold);
    parfor i=1:length(strip_Alpha)
        warning off;
        alpha = strip_Alpha(i);
        width = strip_Width(i);

        sigma = exp(-option.K(SemiIdx, SemiIdx)/width);
        sigma = sqrt(sigma);
        LapFeature = getLapFeature(SemiFeature, sigma, knn);

        % cross validation
        cv_mse = zeros(nfold, 1);
        for k = 1:nfold
            itest = (indices==k); itrain = ~itest;

            X = [LabelFeature(itrain,:); LapFeature];
            Y = [LabelTruth(itrain); zeros(size(LapFeature, 1), 1)];
            [beta, fitInfo] = lasso(X, Y, 'Alpha', alpha, 'Lambda', lambdaRegSet);
            [~, id] = min(fitInfo.MSE);
            predict = LabelFeature(itest,:) * beta(:,id);
            cv_mse(k) = mean((predict - LabelTruth(itest)).^2,1);
        end
        mse_array(i) = mean(cv_mse);
    end
    [~, id] = min(mse_array);
    opt_alpha = strip_Alpha(id);
    opt_width = strip_Width(id);
    sigma = exp(-option.K(SemiIdx, SemiIdx)/opt_width);
    sigma = sqrt(sigma);
    LapFeature = getLapFeature(SemiFeature, sigma, knn);

    X = [LabelFeature; LapFeature];
    Y = [LabelTruth; zeros(size(LapFeature, 1), 1)];
    [beta, fitInfo] = lasso(X, Y, 'Alpha', opt_alpha, 'Lambda', lambdaRegSet);
    [~, id] = min(fitInfo.MSE);
    predict = Test.Feature * beta(:,id) + Ymean;
    res = predict - Test.Truth;

    mse = mean(res.^2);
    mae = mean(abs(res));
    [nLabel opt_alpha opt_width mse mae]
end