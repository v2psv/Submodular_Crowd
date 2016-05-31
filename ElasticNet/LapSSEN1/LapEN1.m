% function [res] = SubWrap_LapEN2(Label, Test, Ymean, SemiFeature, option)
%     if nargin < 6
%         nRepeat = 10;
%     else
%         nRepeat = option.nRepeat;
%     end

%     mae_table = zeros(nRepeat, 1);
%     mse_table = zeros(nRepeat, 1);
%     for iRep = 1:nRepeat
%         opt_beta = LapEN1(Label, SemiFeature, option);
%         predict_raw = Test.Feature * opt_beta' + Ymean;
%         predict = max(round(predict_raw),0);
%         mae_table(iRep) = mean(abs(predict - Test.Truth));
%         mse_table(iRep) = mean((predict - Test.Truth).^2);
%     end

%     res.mae_mean = mean(mae_table); res.mae_std = std(mae_table);
%     res.mse_mean = mean(mse_table); res.mse_std = std(mse_table);
% end

% function [opt_beta] = LapEN1(Label, Test, Ymean, SemiFeature, option)
%     if nargin < 3
%         lambda2Set = [0:0.1:1];
%         lambda3Set = [0:0.1:1];
%         strip_lambda2Set = [];
%         strip_lambda3Set = [];
%         for lambda2 = lambda2Set
%             for lambda3 = lambda3Set
%                 strip_lambda2Set = [strip_lambda2Set; lambda2];
%                 strip_lambda3Set = [strip_lambda3Set; lambda3];
%             end
%         end
%         nfold = 5;
%         nRepeat = 10;
%     else
%         strip_lambda2Set = option.strip_lambda2Set;
%         strip_lambda3Set = option.strip_lambda3Set;
%         nfold = option.nfold;
%         nRepeat = option.nRepeat;
%     end

%     nLabel = length(LabelTruth);
%     mse_array = zeros(length(strip_lambda2Set), nRepeat);
%     mae_array = zeros(length(strip_lambda2Set), nRepeat);

%     parfor iLambda=1:length(strip_lambda2Set)
%         warning off;
%         lambda2 = strip_lambda2Set(iLambda);
%         lambda3 = strip_lambda3Set(iLambda);

%         if option.tuneMethod == 1
%         else if option.tuneMethod == 2
%         else if option.tuneMethod == 3
%         end

%         for iRep = 1:nRepeat
%             opt_beta = cv_lapEN(Label, SemiFeature, lambda2, lambda3, option);
%             predict_raw = Test.Feature * opt_beta' + Ymean;
%             predict = max(round(predict_raw),0);
%             mae_table(iLambda, iRep) = mean(abs(predict - Test.Truth));
%             mse_table(iLambda, iRep) = mean((predict - Test.Truth).^2);
%         end


%         % % n fold cross validation
%         % indices = crossvalind('Kfold',nLabel,nfold);
%         % for k = 1:nfold
%         %     itest = (indices==k); itrain = ~itest;

%         %     X = [Label.Feature(itrain,:); sqrt(lambda3)*SemiFeature];
%         %     Y = [Label.Truth(itrain); zeros(size(SemiFeature, 1), 1)];
%         %     all_beta = larsen(X, Y, lambda2, 0, 0);

%         %     predict = Label.Feature(itest,:) * all_beta';
%         %     res = predict-repmat(Label.Truth(itest,:),1,size(predict,2));
%         %     [min_err, ~] = min(mean(res.^2,1));
%         %     mse_array(i) = mse_array(i) + min_err;
%         % end
%     end

%     [~, id] = min(mse_array);
%     opt_lambda2 = strip_lambda2Set(id);
%     opt_lambda3 = strip_lambda3Set(id);

%     X = [Label.Feature; sqrt(opt_lambda3)*SemiFeature];
%     Y = [Label.Truth; zeros(size(SemiFeature, 1), 1)];
%     all_beta = larsen(X, Y, opt_lambda2, 0, 0);
%     predict = Label.Feature * all_beta';
%     res = predict-repmat(Label.Truth,1,size(predict,2));
%     [~, idx] = min(mean(res.^2, 1));
%     opt_beta = all_beta(idx,:);
% end

% function cv_lapEN(Label, SemiFeature, lambda2, lambda3, option)
%     if nargin < 5
%         nRepeat = 10;
%         nfold = 10;
%     else
%         nRepeat = option.nRepeat;
%         nfold = option.nfold;
%     end

%     nLabel = length(LabelTruth);
%     mse_array = zeros(nfold, 1);

%     % cross validation
%     indices = crossvalind('Kfold', nLabel, nfold);
%     for k = 1:nfold
%         itest = (indices==k); itrain = ~itest;

%         X = [Label.Feature(itrain,:); sqrt(lambda3)*SemiFeature];
%         Y = [Label.Truth(itrain); zeros(size(SemiFeature, 1), 1)];
%         all_beta = larsen(X, Y, lambda2, 0, 0);

%         predict = Label.Feature(itest,:) * all_beta';
%         res = predict - repmat(Label.Truth(itest,:),1,size(predict,2));
%         [min_err, ~] = min(mean(res.^2,1));
%         mse_array(k) = mse_array(k) + min_err;
%     end
% end

function [result] = LapEN1(LabelFeature, LabelTruth, Test, Ymean, SemiFeature, option)
    if nargin < 5
        alphaSet = [0.1 0.5 1];
        lambdaRegSet = [0 0.01 0.1 1 10 100];
        lambdaLapSet = [0:0.2:1];
        strip_alphaSet = [];
        strip_lambdaLapSet = [];
        for alpha = alphaSet
            for lambdaLap = lambdaLapSet
                strip_alphaSet = [strip_alphaSet; alpha];
                strip_lambdaLapSet = [strip_lambdaLapSet; lambdaLap];
            end
        end
        nfold = 10;
    else
        alphaSet = option.alphaSet;
        lambdaRegSet = option.lambdaRegSet;
        lambdaLapSet = option.lambdaLapSet;
        strip_alphaSet = option.strip_alphaSet;
        strip_lambdaLapSet = option.strip_lambdaLapSet;
        nfold = option.nfold;
    end

    nLabel = length(LabelTruth);
    mse_array = zeros(length(strip_alphaSet),length(lambdaRegSet));
    mae_array = zeros(length(strip_alphaSet),length(lambdaRegSet));

    parfor i=1:length(strip_alphaSet)
        warning off;
        alpha = strip_alphaSet(i);
        lambdaLap = strip_lambdaLapSet(i);

        X = [LabelFeature; sqrt(lambdaLap)*SemiFeature];
        Y = [LabelTruth; zeros(size(SemiFeature, 1), 1)];

        [B, ~] = lassoglm(X, Y, 'normal', 'Alpha', alpha, 'CV', nfold, 'Lambda', lambdaRegSet);
        predict = Test.Feature * B + Ymean;
        res = predict - repmat(Test.Truth,1,size(predict,2));
        mse_array(i,:) = mean(res.^2, 1);
        mae_array(i,:) = mean(abs(res), 1);
    end
    result.mse_array = mse_array;
    result.mae_array = mae_array;
end