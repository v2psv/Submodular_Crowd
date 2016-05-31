
function [result] = Wrap_rand_LapSSEN3(Train, Test, option)
	if nargin < 3
		error('In Wrap_rand_LapSSEN:Too few parameters!');
	else
		nRepeat = option.nRepeat;
		LabelNum = option.LabelNum;
		UnlabelNum = option.UnlabelNum;
	end

	result.mse_mean = zeros(length(LabelNum),length(UnlabelNum)); result.mse_std = zeros(length(LabelNum),length(UnlabelNum));
	result.mae_mean = zeros(length(LabelNum),length(UnlabelNum)); result.mae_std = zeros(length(LabelNum),length(UnlabelNum));

	% info.subsetIndex = cell(length(LabelNum),length(UnlabelNum));
	% info.mse = cell(length(LabelNum),length(UnlabelNum));
	% info.mae = cell(length(LabelNum),length(UnlabelNum));

	for iLabel = 1:length(LabelNum)
	    for iUnlabel = 1:length(UnlabelNum)
	        nLabel = LabelNum(iLabel);
	        nUnlabel = min(UnlabelNum(iUnlabel), size(Train.Feature,1)-nLabel);
	        [nLabel nUnlabel]

			mse = zeros(nRepeat, 1);
	        mae = zeros(nRepeat, 1);
	        subsetIndex = zeros(nRepeat, nLabel);

	        parfor iRep = 1:nRepeat
	        	index = randperm(length(Train.Truth));
	        	LabelIdx = index(1:nLabel);
	        	SemiIdx = index(1:nLabel+nUnlabel);

	        	LabelFeature = Train.Feature(LabelIdx, :);
	        	LabelTruth = Train.Truth(LabelIdx, :);
		        Ymean = mean(LabelTruth);
		        LabelTruth = LabelTruth - Ymean;
		        SemiFeature = Train.Feature(SemiIdx,:);

	        	[mse(iRep), mae(iRep)] = CV_LapSSEN3(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
	        	subsetIndex(iRep,:) = LabelIdx;
		    end
		    info.subsetIndex{iLabel,iUnlabel} = subsetIndex;
		    info.mse{iLabel,iUnlabel} = mse;
		    info.mae{iLabel,iUnlabel} = mae;

		    [mean(mse) mean(mae) min(mse) min(mae)]

		    result.mse_mean(iLabel,iUnlabel) = mean(mse); result.mse_std(iLabel,iUnlabel) = std(mse);
		    result.mae_mean(iLabel,iUnlabel) = mean(mae); result.mae_std(iLabel,iUnlabel) = std(mae);
	    end
	end
end

function [mse, mae] = CV_LapSSEN3(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option)
	if nargin < 7
		error('In CV_LapSSEN:Too few parameters!');
	else
		nfold = option.nfold;
		lambdaRegSet = option.lambdaRegSet;
		lambdaLapSet = option.lambdaLapSet;
		widthSet = option.widthSet;
		knn = option.KNN;
	end

	nLabel = size(LabelFeature, 1);
	mse_array = zeros(length(lambdaRegSet)*length(lambdaLapSet)*length(widthSet),1);

	strip_LambdaLap = [];
	strip_LambdaReg = [];
	strip_Width = [];
	for lambdaLap = lambdaLapSet
		for lambdaReg = lambdaRegSet
			for width = widthSet
				strip_LambdaLap = [strip_LambdaLap; lambdaLap];
				strip_LambdaReg = [strip_LambdaReg; lambdaReg];
				strip_Width = [strip_Width; width];
			end
		end
	end

	parfor i=1:length(strip_LambdaReg)
		warning off;
		lambdaLap = strip_LambdaLap(i);
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
	opt_lambdaLap = strip_LambdaLap(id);
	opt_lambdaReg = strip_LambdaReg(id);
	opt_width = strip_Width(id);
	sigma = exp(-option.K(SemiIdx, SemiIdx)/width);
	sigma = sqrt(sigma);
	LapFeature = getLapFeature(SemiFeature, sigma, knn);

	X = [LabelFeature; sqrt(opt_lambdaLap)*LapFeature];
	Y = [LabelTruth; zeros(size(LapFeature, 1), 1)];
	beta = larsen(X, Y, opt_lambdaReg, 0, 0);
	predict = Test.Feature * beta' + Ymean;
	res = predict - repmat(Test.Truth,1,size(predict,2));
	[~, idx] = min(mean(res.^2, 1));

	mse = mean(res(:,idx).^2, 1);
	mae = mean(abs(res(:,idx)), 1);
end