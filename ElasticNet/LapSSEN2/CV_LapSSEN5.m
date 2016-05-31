% CV larsen, both spatial and temporal regularization
function [mse, mae] = CV_LapSSEN5(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option)
	if nargin < 7
		error('In CV_LapSSEN:Too few parameters!');
	else
		nfold = min(option.nfold, size(LabelFeature, 1));
		lambdaRegSet = option.lambdaRegSet;
		widthSset = option.widthSset;
		widthTset = option.widthTset;
		nStep = 1000;
		step = 1/(nStep - 1);
	end

	strip_LambdaReg = [];
	strip_WidthS = [];
	strip_WidthT = [];
	for lambdaReg = lambdaRegSet
		for widthS = widthSset
			for widthT = widthTset
				strip_LambdaReg = [strip_LambdaReg; lambdaReg];
				strip_WidthS = [strip_WidthS; widthS];
				strip_WidthT = [strip_WidthT; widthT];
			end
		end
	end

	[nLabel p] = size(LabelFeature);
	mse_array = zeros(length(strip_LambdaReg),1);
	s_array = zeros(length(strip_LambdaReg),1);
	beta_array = zeros(length(strip_LambdaReg), p);

	indices = crossvalind('Kfold', nLabel, nfold); % warning: fixed CV set for each trial
	parfor i=1:length(strip_LambdaReg)
		warning off;
		lambdaReg = strip_LambdaReg(i);
		widthS = strip_WidthS(i);
		widthT = strip_WidthT(i);

		% sigma = - option.K(SemiIdx, SemiIdx)/widthS - option.T(SemiIdx, SemiIdx)/widthT;
		% sigma = sqrt(exp(sigma));

		% sigmaS = sqrt(exp(-option.K(SemiIdx, SemiIdx)/widthS));
		% sigmaT = sqrt(exp(-option.T(SemiIdx, SemiIdx)/widthT));
		% sigma = sigmaS + sigmaT;

		sigmaS = exp(-option.K(SemiIdx, SemiIdx)/widthS);
		sigmaT = exp(-option.T(SemiIdx, SemiIdx)/widthT);
		sigma = sqrt(sigmaS + sigmaT);

		LapFeature = getLapFeature(SemiFeature, sigma);

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
	opt_widthS = strip_WidthS(id);
	opt_widthT = strip_WidthT(id);
	opt_beta = beta_array(id,:);

	predict = Test.Feature * opt_beta' + Ymean;
	res = predict - Test.Truth;

	mse = mean(res.^2);
	mae = mean(abs(res));
	% [opt_s opt_lambdaReg opt_widthS opt_widthT mse mae]
end