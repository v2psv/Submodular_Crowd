% function [mse, mae] = CV_LapSSEN(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option)
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

% CV larsen, only spatial regularization
function [mse, mae] = CV_LapSSEN1(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option)
	if nargin < 7
		error('In CV_LapSSEN:Too few parameters!');
	else
		nfold = min(option.nfold, size(LabelFeature, 1));
		lambdaRegSet = option.lambdaRegSet;
		widthSset = option.widthSset;
		% knn = option.KNN;
		nStep = 1000;
		step = 1/(nStep - 1);
	end

	strip_LambdaReg = [];
	strip_WidthS = [];
	for lambdaReg = lambdaRegSet
		for widthS = widthSset
			strip_LambdaReg = [strip_LambdaReg; lambdaReg];
			strip_WidthS = [strip_WidthS; widthS];
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
		sigmaS = exp(-option.K(SemiIdx, SemiIdx)/widthS);
		sigmaS = sqrt(sigmaS);
		LapFeature = getLapFeature(SemiFeature, sigmaS);

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
	opt_beta = beta_array(id,:);

	predict = Test.Feature * opt_beta' + Ymean;
	res = predict - Test.Truth;

	mse = mean(res.^2);
	mae = mean(abs(res));
	% [opt_s opt_lambdaReg opt_widthS mse mae]
end