function [ rmse, sde, bias ] = computeErrors( labels, predict )
%computes the rmse, sde and bias errors for two arrays of labels and
%predictions
  
  %%Ignore nights
  validLabels=labels(labels~=0);
  validPredict=predict(labels~=0);
  
  dif = (validLabels-validPredict);
  meanLabels = mean(validLabels);
  
  rmse = sqrt(mean(dif.^2));
  bias = mean(dif)/meanLabels;
  sde  = sqrt(rmse(end)^2 - bias(end)^2)/meanLabels;

end

