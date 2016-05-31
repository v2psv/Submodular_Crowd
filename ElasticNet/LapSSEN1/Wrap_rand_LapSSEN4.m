function [result] = Wrap_rand_LapSSEN4(Train, Test, option)
	if nargin < 3
		error('In Wrap_rand_LapSSEN:Too few parameters!');
	else
		nRepeat = option.nRepeat;
		LabelNum = option.LabelNum;
		UnlabelNum = option.UnlabelNum;
	end

	result.mse_mean = zeros(length(LabelNum),length(UnlabelNum)); result.mse_std = zeros(length(LabelNum),length(UnlabelNum));
	result.mae_mean = zeros(length(LabelNum),length(UnlabelNum)); result.mae_std = zeros(length(LabelNum),length(UnlabelNum));
	result.mse_array = zeros(nRepeat,length(LabelNum),length(UnlabelNum));
	result.mae_array = zeros(nRepeat,length(LabelNum),length(UnlabelNum));
	MSE = cell(1,nRepeat);
	MAE = cell(1,nRepeat);

	for iRep = 1:nRepeat
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

		    	[mse_array(iLabel,iUnlabel), mae_array(iLabel,iUnlabel)] = CV_LapSSEN4_4(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
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

	% for iLabel = 1:length(LabelNum)
	%     for iUnlabel = 1:length(UnlabelNum)
	%         nLabel = LabelNum(iLabel);
	%         nUnlabel = min(UnlabelNum(iUnlabel), size(Train.Feature,1)-nLabel);
	%         [nLabel nUnlabel]

	% 		mse = zeros(nRepeat, 1);
	%         mae = zeros(nRepeat, 1);

	%         parfor iRep = 1:nRepeat
	%         	index = randperm(length(Train.Truth));
	%         	LabelIdx = index(1:nLabel);
	%         	SemiIdx = index(1:nLabel+nUnlabel);

	%         	LabelFeature = Train.Feature(LabelIdx, :);
	%         	LabelTruth = Train.Truth(LabelIdx, :);
	% 	        Ymean = mean(LabelTruth);
	% 	        LabelTruth = LabelTruth - Ymean;
	% 	        SemiFeature = Train.Feature(SemiIdx,:);

	%         	[mse(iRep), mae(iRep)] = CV_LapSSEN4(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
	% 	    end

	% 	    [mean(mse) mean(mae) min(mse) min(mae)]

	% 	    result.mse_mean(iLabel,iUnlabel) = mean(mse); result.mse_std(iLabel,iUnlabel) = std(mse);
	% 	    result.mae_mean(iLabel,iUnlabel) = mean(mae); result.mae_std(iLabel,iUnlabel) = std(mae);
	%     end
	% end
end


% function [mse, mae] = CV_LapSSEN4(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option)
% 	if nargin < 7
% 		error('In CV_LapSSEN:Too few parameters!');
% 	else
% 		nfold = min(option.nfold, size(LabelFeature, 1));
% 		lambdaRegSet = option.lambdaRegSet;
% 		widthSet = option.widthSet;
% 		knn = option.KNN;
% 	end

% 	strip_LambdaReg = [];
% 	strip_Width = [];
% 	for lambdaReg = lambdaRegSet
% 		for width = widthSet
% 			strip_LambdaReg = [strip_LambdaReg; lambdaReg];
% 			strip_Width = [strip_Width; width];
% 		end
% 	end

% 	nLabel = size(LabelFeature, 1);
% 	mse_array = zeros(length(lambdaRegSet)*length(widthSet),1);

% 	parfor i=1:length(strip_LambdaReg)
% 		warning off;
% 		lambdaReg = strip_LambdaReg(i);
% 		width = strip_Width(i);
% 		sigma = exp(-option.K(SemiIdx, SemiIdx)/width);
% 		sigma = sqrt(sigma);
% 		LapFeature = getLapFeature(SemiFeature, sigma, knn);

% 	    % cross validation
% 	    cv_mse = zeros(nfold, 1);
% 	    indices = crossvalind('Kfold', nLabel, nfold);
% 	    for k = 1:nfold
% 	        itest = (indices==k); itrain = ~itest;

% 	        X = [LabelFeature(itrain,:); LapFeature];
% 	        Y = [LabelTruth(itrain); zeros(size(LapFeature, 1), 1)];
% 	        beta = larsen(X, Y, lambdaReg, 0, 0);
% 	        predict = LabelFeature(itest,:) * beta';
% 	        res = predict - repmat(LabelTruth(itest),1,size(predict,2));
% 	        [min_err, ~] = min(mean(res.^2,1));
% 	        cv_mse(k) = min_err;
% 	    end
% 	    mse_array(i) = mean(cv_mse);
% 	end
% 	[~, id] = min(mse_array);
% 	opt_lambdaReg = strip_LambdaReg(id);
% 	opt_width = strip_Width(id);
% 	sigma = exp(-option.K(SemiIdx, SemiIdx)/opt_width);
% 	sigma = sqrt(sigma);
% 	LapFeature = getLapFeature(SemiFeature, sigma, knn);

% 	X = [LabelFeature; LapFeature];
% 	Y = [LabelTruth; zeros(size(LapFeature, 1), 1)];
% 	beta = larsen(X, Y, opt_lambdaReg, 0, 0);
% 	predict = Test.Feature * beta' + Ymean;
% 	res = predict - repmat(Test.Truth,1,size(predict,2));
% 	[~, idx] = min(mean(res.^2, 1));

% 	mse = mean(res(:,idx).^2, 1);
% 	mae = mean(abs(res(:,idx)), 1);
% 	% [opt_lambdaReg opt_width mse mae]
% end

% function [mse, mae] = CV_LapSSEN4_2(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option)
% 	if nargin < 7
% 		error('In CV_LapSSEN:Too few parameters!');
% 	else
% 		nfold = min(option.nfold, size(LabelFeature, 1));
% 		lambdaRegSet = option.lambdaRegSet;
% 		alphaSet = option.alphaSet;
% 		widthSet = option.widthSet;
% 		knn = option.KNN;
% 	end

% 	strip_Alpha = [];
% 	strip_Width = [];
% 	for alpha = alphaSet
% 		for width = widthSet
% 			strip_Alpha = [strip_Alpha; alpha];
% 			strip_Width = [strip_Width; width];
% 		end
% 	end

% 	nLabel = size(LabelFeature, 1);
% 	mse_array = zeros(length(alphaSet)*length(widthSet),1);

% 	parfor i=1:length(strip_Alpha)
% 		warning off;
% 		alpha = strip_Alpha(i);
% 		width = strip_Width(i);

% 		sigma = exp(-option.K(SemiIdx, SemiIdx)/width);
% 		sigma = sqrt(sigma);
% 		LapFeature = getLapFeature(SemiFeature, sigma, knn);

% 	    % cross validation
% 	    cv_mse = zeros(nfold, 1);
% 	    indices = crossvalind('Kfold', nLabel, nfold);
% 	    for k = 1:nfold
% 	        itest = (indices==k); itrain = ~itest;

% 	        X = [LabelFeature(itrain,:); LapFeature];
% 	        Y = [LabelTruth(itrain); zeros(size(LapFeature, 1), 1)];
% 	        [beta, fitInfo] = lasso(X, Y, 'Alpha', alpha, 'Lambda', lambdaRegSet);
% 	        predict = LabelFeature(itest,:) * beta;
% 	        res = predict - repmat(LabelTruth(itest),1,size(predict,2));
% 	        [min_err, ~] = min(mean(res.^2,1));
% 	        cv_mse(k) = min_err;
% 	    end
% 	    mse_array(i) = mean(cv_mse);
% 	end
% 	[~, id] = min(mse_array);
% 	opt_alpha = strip_Alpha(id);
% 	opt_width = strip_Width(id);
% 	sigma = exp(-option.K(SemiIdx, SemiIdx)/opt_width);
% 	sigma = sqrt(sigma);
% 	LapFeature = getLapFeature(SemiFeature, sigma, knn);

% 	X = [LabelFeature; LapFeature];
% 	Y = [LabelTruth; zeros(size(LapFeature, 1), 1)];
% 	[beta, fitInfo] = lasso(X, Y, 'Alpha', opt_alpha, 'Lambda', lambdaRegSet);
% 	[~, id] = min(fitInfo.MSE);
% 	predict = Test.Feature * beta(:,id) + Ymean;
% 	res = predict - Test.Truth;

% 	mse = mean(res.^2);
% 	mae = mean(abs(res));
% 	[opt_alpha opt_width mse mae]
% end

function [mse, mae] = CV_LapSSEN4_3(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option)
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
	[opt_alpha opt_width mse mae]
end

function [mse, mae] = CV_LapSSEN4_4(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option)
	if nargin < 7
		error('In CV_LapSSEN:Too few parameters!');
	else
		nfold = min(option.nfold, size(LabelFeature, 1));
		lambdaRegSet = option.lambdaRegSet;
		widthSet = option.widthSet;
		knn = option.KNN;
		nStep = 1000;
		step = 1/(nStep - 1);
	end

	strip_LambdaReg = [];
	strip_Width = [];
	for lambdaReg = lambdaRegSet
		for width = widthSet
			strip_LambdaReg = [strip_LambdaReg; lambdaReg];
			strip_Width = [strip_Width; width];
		end
	end

	[nLabel p] = size(LabelFeature);
	mse_array = zeros(length(lambdaRegSet)*length(widthSet),1);
	s_array = zeros(length(lambdaRegSet)*length(widthSet),1);
	beta_array = zeros(length(lambdaRegSet)*length(widthSet), p);

	indices = crossvalind('Kfold', nLabel, nfold); % warning: fixed CV set for each trial
	parfor i=1:length(strip_LambdaReg)
		warning off;
		lambdaReg = strip_LambdaReg(i);
		width = strip_Width(i);
		sigma = exp(-option.K(SemiIdx, SemiIdx)/width);
		sigma = sqrt(sigma);
		LapFeature = getLapFeature(SemiFeature, sigma, knn);

	    % cross validation
	    res = zeros(nfold, nStep);
	    cv_mse = zeros(nfold, 1);
	    for k = 1:nfold
	        itest = (indices==k); itrain = ~itest;
	        X = [LabelFeature(itrain,:); LapFeature];
	        Y = [LabelTruth(itrain); zeros(size(LapFeature, 1), 1)];

	        beta = larsen(X, Y, lambdaReg, 0, 0);
	        t = sum(abs(beta),2);
	        s = (t - min(t))/max(t - min(t));
	        [sm s_idx] = unique(s, 'rows');
	        beta_interp = interp1q(s(s_idx), beta(s_idx, :), (0:step:1)');
	        res(k, :) = sum((LabelTruth(itest)*ones(1,nStep) - LabelFeature(itest,:)*beta_interp').^2);
	    end
	    res_mean = mean(res); res_std = std(res);
	    [res_min idx_opt] = min(res_mean);

	    %% Find optimal coefficient vector
	    s_opt = idx_opt/nStep;
	    beta = larsen(X, Y, lambdaReg, 0, 0);
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
	opt_lambdaReg = strip_LambdaReg(id);
	opt_width = strip_Width(id);
	opt_beta = beta_array(id,:);

	sigma = exp(-option.K(SemiIdx, SemiIdx)/opt_width);
	sigma = sqrt(sigma);
	LapFeature = getLapFeature(SemiFeature, sigma, knn);

	% X = [LabelFeature; LapFeature];
	% Y = [LabelTruth; zeros(size(LapFeature, 1), 1)];
	% [beta, fitInfo] = lasso(X, Y, 'Alpha', opt_alpha, 'Lambda', lambdaRegSet);
	% [~, id] = min(fitInfo.MSE);
	% predict = Test.Feature * beta(:,id) + Ymean;
	predict = Test.Feature * opt_beta' + Ymean;
	res = predict - Test.Truth;

	mse = mean(res.^2);
	mae = mean(abs(res));
	[opt_s opt_lambdaReg opt_width mse mae]
end