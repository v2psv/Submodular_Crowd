function [result] = Wrap_LapSSEN3(Train, Test, option)
    if nargin < 3
        error('In Wrap_LapEN(Feature, Truth, option):Too few parameters!');
    end
    knn = option.KNN;
    MSE = cell(1, option.nRepeat);
    MAE = cell(1, option.nRepeat);

    parfor iRep = 1:option.nRepeat
        index = randperm(length(Train.Truth));
        TrainIndex = sort(index(1:option.TrainNum));
        TrainFeature = Train.Feature(TrainIndex,:);
        TrainTruth = Train.Truth(TrainIndex);
        LapFeature = getLapFeature(TrainFeature, option.AM(TrainIndex,TrainIndex), knn);

        % subsetIndex: selected subset indices of TrainFeature
        if strcmp(option.term, 'temporal')==1
            Ntrain = [1:option.TrainNum];
            T = TrainIndex';
            sigma = pdist2(T, T).^2;
            F = sub_fn_facility(sigma, Ntrain);
            [subsetIndex, scores] = greedy(F, Ntrain, option.bound, option.init_S);
        end
        if strcmp(option.term, 'random')==1
            index = randperm(option.TrainNum);
            subsetIndex = index(1:option.bound);
        end

        mse_array = zeros(length(option.LabelNum),1);
        mae_array = zeros(length(option.LabelNum),1);
        for i = 1:length(option.LabelNum)
            nLabel = option.LabelNum(i);
            LabelIdx = subsetIndex(1:nLabel);
            LabelFeature = TrainFeature(LabelIdx, :);
            LabelTruth = TrainTruth(LabelIdx, :);
            Ymean = mean(LabelTruth);
            LabelTruth = LabelTruth - Ymean;

            [mse_array(i), mae_array(i)] = CV_LapSSEN3(LabelFeature, LabelTruth, LapFeature, Ymean, Test, option);
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

function [mse, mae] = CV_LapSSEN3(LabelFeature, LabelTruth, LapFeature, Ymean, Test, option)
    if nargin < 6
        error('In CV_LapSSEN:Too few parameters!');
    else
        nLabel = size(LabelFeature, 1);
        nfold = min(option.nfold, nLabel);
        lambdaRegSet = option.lambdaRegSet;
        lambdaLapSet = option.lambdaLapSet;
    end

    % stripping 2D parameters for parallel computing
    strip_LambdaReg = [];
    strip_LambdaLap = [];
    for lambdaReg = lambdaRegSet
        for lambdaLap = lambdaLapSet
            strip_LambdaReg = [strip_LambdaReg; lambdaReg];
            strip_LambdaLap = [strip_LambdaLap; lambdaLap];
        end
    end

    mse_array = zeros(length(lambdaRegSet)*length(lambdaLapSet),1);
    parfor i=1:length(strip_LambdaReg)
        warning off;
        lambdaReg = strip_LambdaReg(i);
        lambdaLap = strip_LambdaLap(i);

        % cross validation
        cv_mse = zeros(nfold, 1);
        indices = crossvalind('Kfold', nLabel, nfold);
        for k = 1:nfold
            itest = (indices==k); itrain = ~itest;
            X = [LabelFeature(itrain,:); sqrt(lambdaLap)*LapFeature];
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
    opt_lambdaLap = strip_LambdaLap(id);

    X = [LabelFeature; sqrt(opt_lambdaLap)*LapFeature];
    Y = [LabelTruth; zeros(size(LapFeature, 1), 1)];
    beta = larsen(X, Y, opt_lambdaReg, 0, 0);
    predict = Test.Feature * beta' + Ymean;
    res = predict - repmat(Test.Truth,1,size(predict,2));
    [~, idx] = min(mean(res.^2, 1));

    mse = mean(res(:,idx).^2, 1);
    mae = mean(abs(res(:,idx)), 1);
    [nLabel opt_lambdaReg opt_lambdaLap mse]
end