% CV larsen, only temporal regularization
function [mse, mae] = CV_LapSSEN2(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option)
	if nargin < 7
		error('In CV_LapSSEN:Too few parameters!');
	else
		nfold = min(option.nfold, size(LabelFeature, 1));
		lambdaRegSet = option.lambdaRegSet;
		widthTset = option.widthTset;
		% ktt = option.KTT;
		nStep = 1000;
		step = 1/(nStep - 1);
	end

	strip_LambdaReg = [];
	strip_WidthT = [];
	for lambdaReg = lambdaRegSet
		for widthT = widthTset
			strip_LambdaReg = [strip_LambdaReg; lambdaReg];
			strip_WidthT = [strip_WidthT; widthT];
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
		widthT = strip_WidthT(i);
		sigmaT = sqrt(exp(-option.T(SemiIdx, SemiIdx)/widthT));
		LapTFeature = getLapFeature(SemiFeature, sigmaT);

	    % cross validation
	    res = zeros(nfold, nStep);
	    cv_mse = zeros(nfold, 1);
	    for k = 1:nfold
	        itest = (indices==k); itrain = ~itest;
	        X = [LabelFeature(itrain,:); LapTFeature];
	        Y = [LabelTruth(itrain); zeros(size(LapTFeature, 1), 1)];

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
	opt_widthT = strip_WidthT(id);
	opt_beta = beta_array(id,:);

	% sigmaT = sqrt(exp(-option.T(SemiIdx, SemiIdx)/opt_widthT));
	% LapTFeature = getLapFeature(SemiFeature, sigmaT);

	predict = Test.Feature * opt_beta' + Ymean;
	res = predict - Test.Truth;

	mse = mean(res.^2);
	mae = mean(abs(res));
	% [opt_s opt_lambdaReg opt_widthT mse mae]
end