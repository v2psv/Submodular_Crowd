% function [result] = Wrap_LapEN(DataIndex, Train, Test, Params, option)
%     if nargin < 5
%         error('In Wrap_LapEN(Feature, Truth, option):Too few parameters!');
%     end

%     nTrainNum = length(option.TrainNum);
%     nIndex = size(DataIndex,1);

%     MSE = cell(nTrainNum, nIndex);
%     MAE = cell(nTrainNum, nIndex);
%     MSE_MIN = zeros(nTrainNum, nIndex);
%     MAE_MIN = zeros(nTrainNum, nIndex);

%     for iTrain = 1:nTrainNum
%         trainNum = option.TrainNum(iTrain)
%         parfor iIndex = 1:nIndex
%             labelSet = DataIndex(iIndex, 1:trainNum);
%             LabelFeature = Train.Feature(labelSet,:);
%             LabelTruth = Train.Truth(labelSet);
%             % normalize Y
%             Ymean = mean(LabelTruth);
%             LabelTruth = LabelTruth - Ymean;

%             res = LapEN1(LabelFeature, LabelTruth, Test, Ymean, Params(iIndex).SemiFeature, option);
%             MSE{iTrain,iIndex} = res.mse_array;
%             MAE{iTrain,iIndex} = res.mae_array;
%             MSE_MIN(iTrain,iIndex) = min(min(res.mse_array));
%             MAE_MIN(iTrain,iIndex) = min(min(res.mae_array));
%         end
%     end
%     result.MSE = MSE;
%     result.MAE = MAE;
%     result.MSE_MIN = MSE_MIN;
%     result.MAE_MIN = MAE_MIN;
% end


function [result] = Wrap_LapEN(DataIndex, Train, Test, sigma, option)
    if nargin < 5
        error('In Wrap_LapEN(Feature, Truth, option):Too few parameters!');
    end
    cluster = option.cluster;
    nTrainNum = length(option.TrainNum);
    nIndex = size(DataIndex,1);

    MSE = zeros(nTrainNum, nIndex);
    MAE = zeros(nTrainNum, nIndex);

    for iTrain = 1:nTrainNum
        trainNum = option.TrainNum(iTrain)
        parfor iIndex = 1:nIndex
            labelSet = DataIndex(iIndex, 1:trainNum);
            LabelFeature = Train.Feature(labelSet,:);
            LapFeature = getLapFeature(Train.Feature, [1:size(Train.Feature,1)], sigma, cluster);
            LabelTruth = Train.Truth(labelSet);
            % normalize Y
            Ymean = mean(LabelTruth);
            LabelTruth = LabelTruth - Ymean;
            [mse, mae] = CV_LapSSEN(LabelFeature, LabelTruth, Ymean, LapFeature, Test, option);
            MSE(iTrain,iIndex) = mse;
            MAE(iTrain,iIndex) = mae;
        end
    end
    result.MSE = MSE;
    result.MAE = MAE;
end



function [mse, mae] = CV_LapSSEN(LabelFeature, LabelTruth, Ymean, LapFeature, Test, option)
    if nargin < 5
        nfold = 5;
    else
        nfold = option.nfold;
        lambdaRegSet = option.lambdaRegSet;
        lambdaLapSet = option.lambdaLapSet;
    end

    nLabel = size(LabelFeature, 1);
    mse_array = zeros(length(lambdaRegSet)*length(lambdaLapSet),1);

    strip_LambdaLap = [];% zeros(length(lambdaRegSet)*length(lambdaLapSet),1);
    strip_LambdaReg = [];% zeros(length(lambdaRegSet)*length(lambdaLapSet),1);
    for lambdaLap = lambdaLapSet
        for lambdaReg = lambdaRegSet
            strip_LambdaLap = [strip_LambdaLap; lambdaLap];
            strip_LambdaReg = [strip_LambdaReg; lambdaReg];
        end
    end

    parfor i=1:length(strip_LambdaReg)
        warning off;
        lambdaLap = strip_LambdaLap(i);
        lambdaReg = strip_LambdaReg(i);

        % cross validation
        cv_mse = zeros(nfold, 1);
        indices = crossvalind('Kfold', nLabel, nfold);
        for k = 1:nfold
            itest = (indices==k); itrain = ~itest;

            X = [LabelFeature(itrain,:); sqrt(lambdaLap)*LapFeature];
            Y = [LabelTruth(itrain); zeros(size(LapFeature, 1), 1)];
            beta = larsen(X, Y, lambdaReg, 0, 0);
            % [B, FitInfo] = lassoglm(X, Y, 'normal', 'Alpha', alpha, 'Lambda', lambdaReg);
            predict = LabelFeature(itest,:) * beta';
            res = predict - repmat(LabelTruth(itest),1,size(predict,2));
            [min_err, ~] = min(mean(res.^2,1));
            cv_mse(k) = min_err;
        end
        mse_array(i) = mean(cv_mse);
    end
    [~, id] = min(mse_array);
    opt_lambdaLap = strip_LambdaLap(id);
    opt_lambdaReg = strip_LambdaReg(id);

    X = [LabelFeature; sqrt(opt_lambdaLap)*LapFeature];
    Y = [LabelTruth; zeros(size(LapFeature, 1), 1)];
    beta = larsen(X, Y, opt_lambdaReg, 0, 0);
    predict = Test.Feature * beta' + Ymean;
    res = predict - repmat(Test.Truth,1,size(predict,2));
    [~, idx] = min(mean(res.^2, 1));

    mse = mean(res(:,idx).^2, 1);
    mae = mean(abs(res(:,idx)), 1);
end