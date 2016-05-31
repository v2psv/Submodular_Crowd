function [result] = Wrap_rand_LapSSEN5(Train, Test, option)
	if nargin < 3
		error('In Wrap_rand_LapSSEN:Too few parameters!');
	else
		knn = option.KNN;
		nRepeat = option.nRepeat;
		LabelNum = option.LabelNum;
		UnlabelNum = option.UnlabelNum;
        option.AM = sqrt(option.AM);
	end

	result.mse_mean = zeros(length(LabelNum),length(UnlabelNum)); result.mse_std = zeros(length(LabelNum),length(UnlabelNum));
	result.mae_mean = zeros(length(LabelNum),length(UnlabelNum)); result.mae_std = zeros(length(LabelNum),length(UnlabelNum));
	result.mse_array = zeros(nRepeat,length(LabelNum),length(UnlabelNum));
	result.mae_array = zeros(nRepeat,length(LabelNum),length(UnlabelNum));
	MSE = cell(1,nRepeat);
	MAE = cell(1,nRepeat);

	parfor iRep = 1:nRepeat
		mse_array = zeros(length(LabelNum),length(UnlabelNum));
		mae_array = zeros(length(LabelNum),length(UnlabelNum));

		index = randperm(length(Train.Truth));

		for iLabel = 1:length(LabelNum)
			nLabel = LabelNum(iLabel);
			LabelIdx = index(1:nLabel);
			LabelFeature = Train.Feature(LabelIdx, :);
			LabelTruth = Train.Truth(LabelIdx, :);
			Ymean = mean(LabelTruth);
			LabelTruth = LabelTruth - Ymean;

		    for iUnlabel = 1:length(UnlabelNum)
		    	nUnlabel = min(UnlabelNum(iUnlabel), length(Train.Truth)-nLabel);
		    	SemiIdx = index(1:nLabel+nUnlabel);
		    	SemiFeature = Train.Feature(SemiIdx,:);
		    	LapFeature = getLapFeature(SemiFeature, option.AM(SemiIdx,SemiIdx), knn);

		    	[mse_array(iLabel,iUnlabel), mae_array(iLabel,iUnlabel)] = CV_LapSSEN5(LabelFeature, LabelTruth, LapFeature, Ymean, Test, option);
		    	[nLabel nUnlabel mse_array(iLabel,iUnlabel) mae_array(iLabel,iUnlabel)]
		    end
		end
		MSE{iRep} = mse_array;
		MAE{iRep} = mae_array;
	end
	for i=1:nRepeat
		result.mse_array(i,:,:) = MSE{i};
		result.mae_array(i,:,:) = MAE{i};
	end
	result.mse_mean = squeeze(mean(result.mse_array,1)); result.mse_std = squeeze(std(result.mse_array,1));
	result.mae_mean = squeeze(mean(result.mae_array,1)); result.mae_std = squeeze(std(result.mae_array,1));
end

function [mse, mae] = CV_LapSSEN5(LabelFeature, LabelTruth, LapFeature, Ymean, Test, option)
    if nargin < 6
        error('In CV_LapSSEN:Too few parameters!');
    else
        nLabel = size(LabelFeature, 1);
        nfold = min(option.nfold, nLabel);
        lambdaRegSet = option.lambdaRegSet;
        lambdaLapSet = option.lambdaLapSet;
        alphaSet = option.alphaSet;
    end

    if size(LapFeature,1)==0
    	lambdaLapSet = [0];
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
    % [nLabel opt_lambdaReg opt_lambdaLap mse]
end