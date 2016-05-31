function [mse, mae] = CV_GPR(Labeled, Test, option)
  gpm = gp_train(Labeled.Feature', Labeled.Label', option.covfunc, option.gpnorm, option.gptrials);
  [Y_raw, Spred] = gp_predict(Test.Feature', gpm);
  Y = max(round(Y_raw), 0);           % truncate and round prediction

  mae = mean(abs(Y - Test.Label));
  mse = mean((Y - Test.Label).^2);
end
