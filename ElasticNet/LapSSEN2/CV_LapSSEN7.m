% similarity: w = (1-theta)*w_s + theta*w_t
% regularization: \lambda_I * w
% CV larsen, both spatial and temporal regularization
function [mse, mae] = CV_LapSSEN7(type, LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option)
	if nargin < 7
		error('In CV_LapSSEN:Too few parameters!');
	else
		nfold = min(option.nfold, size(LabelFeature, 1));
		lambdaASet = option.lambdaASet;
		lambdaISet = option.lambdaISet;
		nStep = 100;
		step = 1/(nStep - 1);
	end

	if strcmp(type, 'N')==1
		lambdaISet = [0];
		thetaSet = [0];
	elseif strcmp(type, 'S')==1
		thetaSet = [0];
	elseif strcmp(type, 'T')==1
		thetaSet = [1];
	elseif strcmp(type, 'ST')==1
		thetaSet = option.thetaSet;
	end

	strip_LambdaASet = [];
	strip_LambdaISet = [];
	strip_ThetaSet = [];
	for lambdaA = lambdaASet
		for lambdaI = lambdaISet
			for theta = thetaSet
				strip_LambdaASet = [strip_LambdaASet; lambdaA];
				strip_LambdaISet = [strip_LambdaISet; lambdaI];
				strip_ThetaSet = [strip_ThetaSet; theta];
			end
		end
	end

	[nLabel p] = size(LabelFeature);
	sigmaS = exp(-option.K(SemiIdx, SemiIdx));
	sigmaT = exp(-option.T(SemiIdx, SemiIdx));

	mse_array = zeros(length(strip_LambdaASet),1);
	beta_array = zeros(length(strip_LambdaASet), p);
	s_array = zeros(length(strip_LambdaASet),1);

	indices = crossvalind('Kfold', nLabel, nfold); % warning: fixed CV set for each trial
	parfor i=1:length(strip_LambdaASet)
		warning off;
		lambdaA = strip_LambdaASet(i);
		lambdaI = strip_LambdaISet(i);
		theta = strip_ThetaSet(i);

		sigma = lambdaI*((1-theta)*sigmaS + theta*sigmaT);
		LapFeature = getLapFeature(SemiFeature, sigma, option.KNN);

	    % cross validation
	    res = zeros(nfold, nStep);
	    cv_mse = zeros(nfold, 1);
	    for k = 1:nfold
	        itest = (indices==k); itrain = ~itest;
	        X = [LabelFeature(itrain,:); LapFeature];
	        Y = [LabelTruth(itrain); zeros(size(LapFeature, 1), 1)];

	        beta = larsen(X, Y, lambdaA, 0, 0);
	        t = sum(abs(beta),2);
	        s = (t - min(t))/max(t - min(t));
	        [sm s_idx] = unique(s, 'rows');
	        beta_interp = interp1q(s(s_idx), beta(s_idx, :), (0:step:1)');
	        res(k, :) = sum((LabelTruth(itest)*ones(1,nStep) - LabelFeature(itest,:)*beta_interp').^2);
	    end
	    res_mean = mean(res); res_std = std(res);
	    [res_min idx_opt] = min(res_mean);

	    % limit = res_min + res_std(idx_opt)/2;
	    % idx_opt2 = find(res_mean < limit, 1);
	    % idx_opt = idx_opt2;

	    %% Find optimal coefficient vector
	    s_opt = idx_opt/nStep;
	    beta = larsen(X, Y, lambdaA, 0, 0);
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
	opt_lambdaA = strip_LambdaASet(id);
	opt_lambdaI = strip_LambdaISet(id);
	opt_theta = strip_ThetaSet(id);
	opt_beta = beta_array(id,:);

	predict = Test.Feature * opt_beta' + Ymean;
	predict = max(round(predict),0);
	res = predict - Test.Truth;

	mse = mean(res.^2);
	mae = mean(abs(res));
	% [opt_s opt_lambdaA opt_lambdaI opt_theta mse mae]
end