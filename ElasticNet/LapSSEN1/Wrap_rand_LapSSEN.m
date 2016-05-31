function [result] = Wrap_rand_LapSSEN(Train, Test, sigma, option)
	if nargin < 4
		error('In Wrap_rand_LapSSEN:Too few parameters!');
	else
		nRepeat = option.nRepeat;
		cluster = option.cluster;
		LabelNum = option.LabelNum;
		UnlabelNum = option.UnlabelNum;
	end

	result.mse_Min_mean = zeros(length(LabelNum),length(UnlabelNum)); result.mse_Min_std = zeros(length(LabelNum),length(UnlabelNum));
	result.mse_1SE_mean = zeros(length(LabelNum),length(UnlabelNum)); result.mse_1SE_std = zeros(length(LabelNum),length(UnlabelNum));
	result.mae_Min_mean = zeros(length(LabelNum),length(UnlabelNum)); result.mae_Min_std = zeros(length(LabelNum),length(UnlabelNum));
	result.mae_1SE_mean = zeros(length(LabelNum),length(UnlabelNum)); result.mae_1SE_std = zeros(length(LabelNum),length(UnlabelNum));

	for iLabel = 1:length(LabelNum)
	    for iUnlabel = 1:length(UnlabelNum)
	        nLabel = LabelNum(iLabel)
	        nUnlabel = UnlabelNum(iUnlabel)

	        mse_Min = zeros(nRepeat, 1); mse_1SE = zeros(nRepeat, 1);
	        mae_Min = zeros(nRepeat, 1); mae_1SE = zeros(nRepeat, 1);
	        parfor iRep = 1:nRepeat
	        	index = randperm(length(Train.Truth));
	        	LabelIdx = index(1:nLabel);
	        	SemiIdx = index(1:nLabel+nUnlabel);

	        	LabelFeature = Train.Feature(LabelIdx, :);
	        	LabelTruth = Train.Truth(LabelIdx, :);
		        Ymean = mean(LabelTruth);
		        LabelTruth = LabelTruth - Ymean;

	        	LapFeature = getLapFeature(Train.Feature, SemiIdx, sigma, cluster);
	        	[mse_Min(iRep), mse_1SE(iRep), mae_Min(iRep), mae_1SE(iRep)] = CV_LapSSEN(LabelFeature, LabelTruth, Ymean, LapFeature, Test, option);
		    end
		    result.mse_Min_mean(iLabel,iUnlabel) = mean(mse_Min); result.mse_Min_std = std(mse_Min);
		    result.mse_1SE_mean(iLabel,iUnlabel) = mean(mse_1SE); result.mse_1SE_std = std(mse_1SE);
		    result.mae_Min_mean(iLabel,iUnlabel) = mean(mae_Min); result.mae_Min_std = std(mae_Min);
		    result.mae_1SE_mean(iLabel,iUnlabel) = mean(mae_1SE); result.mae_1SE_std = std(mae_1SE);
		    result
	    end
	end
end

function [mse_Min, mse_1SE, mae_Min, mae_1SE] = CV_LapSSEN(LabelFeature, LabelTruth, Ymean, LapFeature, Test, option)
	if nargin < 5
		nfold = 10;
		alpha = 1;	% lasso
	else
		nfold = option.nfold;
		alpha = option.alpha;
		lambdaRegSet = option.lambdaRegSet;
	end

	% X = [LabelFeature; sqrt(lambdaLap)*LapFeature];
	X = [LabelFeature; LapFeature];
	Y = [LabelTruth; zeros(size(LapFeature, 1), 1)];

	opt = statset('UseParallel', true);
	% [B, FitInfo] = lassoglm(X, Y, 'normal', 'CV', nfold, 'Alpha', alpha, 'Lambda', lambdaRegSet, 'Options', opt);
	[B, FitInfo] = lassoglm(X, Y, 'normal', 'CV', nfold, 'Alpha', alpha, 'Options', opt);
	beta_Min = B(:,FitInfo.IndexMinDeviance);
	beta_1SE = B(:,FitInfo.Index1SE);

	predict = Test.Feature * beta_Min + Ymean;
	err = predict - repmat(Test.Truth,1,size(predict,2));
	mse_Min = mean(err.^2, 1);
	mae_Min = mean(abs(err), 1);

	predict = Test.Feature * beta_1SE + Ymean;
	err = predict - repmat(Test.Truth,1,size(predict,2));
	mse_1SE = mean(err.^2, 1);
	mae_1SE = mean(abs(err), 1);
end