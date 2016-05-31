% function [result] = Wrap_rand_LapSSEN(Train, Test, option)
% 	if nargin < 3
% 		error('In Wrap_rand_LapSSEN:Too few parameters!');
% 	else
% 		nRepeat = option.nRepeat;
% 		LabelNum = option.LabelNum;
% 		UnlabelNum = option.UnlabelNum;
% 	end

% 	result.mse_mean = zeros(length(LabelNum),length(UnlabelNum)); result.mse_std = zeros(length(LabelNum),length(UnlabelNum));
% 	result.mae_mean = zeros(length(LabelNum),length(UnlabelNum)); result.mae_std = zeros(length(LabelNum),length(UnlabelNum));
% 	result.mse_array = zeros(nRepeat,length(LabelNum),length(UnlabelNum));
% 	result.mae_array = zeros(nRepeat,length(LabelNum),length(UnlabelNum));
% 	MSE = cell(1,nRepeat);
% 	MAE = cell(1,nRepeat);

% 	for iRep = 1:nRepeat
% 		mse_array = zeros(length(LabelNum),length(UnlabelNum));
% 		mae_array = zeros(length(LabelNum),length(UnlabelNum));

% 		index = randperm(length(Train.Truth));

% 		for iLabel = 1:length(LabelNum)
% 			nLabel = LabelNum(iLabel);
% 			LabelIdx = index(1:nLabel);
% 			LabelFeature = Train.Feature(LabelIdx, :);
% 			LabelTruth = Train.Truth(LabelIdx, :);
% 			Ymean = mean(LabelTruth);
% 			LabelTruth = LabelTruth - Ymean;

% 		    for iUnlabel = 1:length(UnlabelNum)
% 		    	nUnlabel = min(UnlabelNum(iUnlabel), length(Train.Truth)-nLabel);
% 		    	SemiIdx = index(1:nLabel+nUnlabel);
% 		    	SemiFeature = Train.Feature(SemiIdx,:);

% 		    	if strcmp(option.kernel, 'S')==1
% 		    		[mse_array(iLabel,iUnlabel), mae_array(iLabel,iUnlabel)] = CV_LapSSEN1(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
% 		    	elseif strcmp(option.kernel, 'T')==1
% 		    		[mse_array(iLabel,iUnlabel), mae_array(iLabel,iUnlabel)] = CV_LapSSEN2(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
% 		    	elseif strcmp(option.kernel, 'ST')==1
% 		    		[mse_array(iLabel,iUnlabel), mae_array(iLabel,iUnlabel)] = CV_LapSSEN7(LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
% 		    	elseif strcmp(option.kernel, 'N')==1
% 		    		[mse_array(iLabel,iUnlabel), mae_array(iLabel,iUnlabel)] = CV_LapSSEN4(LabelFeature, LabelTruth, Ymean, Test, option);
% 		    	end
% 		    	[nLabel nUnlabel mse_array(iLabel,iUnlabel) mae_array(iLabel,iUnlabel)]
% 		    end
% 		end
% 		MSE{iRep} = mse_array;
% 		MAE{iRep} = mae_array;
% 	end
% 	for i=1:nRepeat
% 		result.mse_array(i,:,:) = MSE{i};
% 		result.mae_array(i,:,:) = MAE{i};
% 	end
% 	result.mse_mean = squeeze(mean(result.mse_array,1)); result.mse_std = squeeze(std(result.mse_array,1));
% 	result.mae_mean = squeeze(mean(result.mae_array,1)); result.mae_std = squeeze(std(result.mae_array,1));
% end

function [result] = Wrap_rand_LapSSEN(Train, Test, option)
	if nargin < 3
		error('In Wrap_rand_LapSSEN:Too few parameters!');
	else
		nRepeat = option.nRepeat;
		LabelNum = option.LabelNum;
		nUnlabel = option.UnlabelNum(1);
	end

	MSE = zeros(4, nRepeat, length(LabelNum));
	MAE = zeros(4, nRepeat, length(LabelNum));

	for iRep = 1:nRepeat

		index = randperm(length(Train.Truth));

		for iLabel = 1:length(LabelNum)
			nLabel = LabelNum(iLabel)
			LabelIdx = index(1:nLabel);
			LabelFeature = Train.Feature(LabelIdx, :);
			LabelTruth = Train.Truth(LabelIdx, :);
			Ymean = mean(LabelTruth);
			LabelTruth = LabelTruth - Ymean;

			SemiIdx = index(1:nLabel+nUnlabel);
			SemiFeature = Train.Feature(SemiIdx,:);

			if strcmp(option.kernel, 'A')==1
				[mse1 mae1] = CV_LapSSEN7('N', LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
				[mse2 mae2] = CV_LapSSEN7('S', LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
				[mse3 mae3] = CV_LapSSEN7('T', LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
				[mse4 mae4] = CV_LapSSEN7('ST', LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
				MSE(1, iRep, iLabel) = mse1; MAE(1, iRep, iLabel) = mae1;
				MSE(2, iRep, iLabel) = mse2; MAE(2, iRep, iLabel) = mae2;
				MSE(3, iRep, iLabel) = mse3; MAE(3, iRep, iLabel) = mae3;
				MSE(4, iRep, iLabel) = mse4; MAE(4, iRep, iLabel) = mae4;
				disp([mse1 mse2 mse3 mse4]);
			elseif strcmp(option.kernel, 'N')==1
				[mse mae] = CV_LapSSEN7('N', LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
				MSE(1, iRep, iLabel) = mse; MAE(1, iRep, iLabel) = mae;
			elseif strcmp(option.kernel, 'S')==1
				[mse mae] = CV_LapSSEN7('S', LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
				MSE(2, iRep, iLabel) = mse; MAE(2, iRep, iLabel) = mae;
			elseif strcmp(option.kernel, 'T')==1
				[mse mae] = CV_LapSSEN7('T', LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
				MSE(3, iRep, iLabel) = mse; MAE(3, iRep, iLabel) = mae;
				disp([mse mae]);
			elseif strcmp(option.kernel, 'ST')==1
				[mse mae] = CV_LapSSEN7('ST', LabelFeature, LabelTruth, SemiFeature, SemiIdx, Ymean, Test, option);
				MSE(4, iRep, iLabel) = mse; MAE(4, iRep, iLabel) = mae;
			end
		end
	end
	result.MSE = MSE;
	result.MAE = MAE;
	result.mse_mean = squeeze(mean(MSE, 2));
	result.mae_mean = squeeze(mean(MAE, 2));
end