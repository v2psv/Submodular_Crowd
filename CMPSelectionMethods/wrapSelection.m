function [mse_array, mae_array] = wrapSelection(Train, Test, methods, regress, option)
    if nargin < 5
        error('In wrap_kmeans:Too few parameters!');
    end

    Pool.Feature = Train.Feature(option.PoolIndex, :);
    Pool.Label = Train.Label(option.PoolIndex);

    mse_array = zeros(1, length(methods));
    mae_array = zeros(1, length(methods));
    for i = 1:length(methods)
      LabeledIdx = SelectFrames(Pool, methods(i), option);

      if strcmp(regress, 'GPR')==1
        Labeled.Feature = option.rawTrain.Feature(LabeledIdx, :);
        Labeled.Label = option.rawTrain.Label(LabeledIdx);
      else
        Labeled.Feature = Train.Feature(LabeledIdx, :);
        Labeled.Label = Train.Label(LabeledIdx);
        Ymean = mean(Labeled.Label);
        Labeled.Label = Labeled.Label - Ymean;
      end

      if strcmp(regress, 'GPR')==1
        [mse_array(i), mae_array(i)] = CV_GPR(Labeled, Test, option);
      elseif strcmp(regress, 'SSR')==1
        SemiFeature = Train.Feature;
        [mse_array(i), mae_array(i)] = CV_SSR(Labeled, SemiFeature, Ymean, Test, option);
      elseif strcmp(regress, 'EN')==1
        [mse_array(i), mae_array(i)] = CV_EN(Labeled, Ymean, Test, option);
      elseif strcmp(regress, 'SSEN')==1
        SemiFeature = Train.Feature;
        [mse_array(i), mae_array(i)] = CV_SSEN(Labeled, SemiFeature, Ymean, Test, option);
      end
    end
end
