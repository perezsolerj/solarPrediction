%%% Evaluates the prediction assuming labels and predict variables are
%%% created and have the shape [instances, predictionInstants]

rmse=[];
for i=1:size(labels,2)
  predLabels = labels(:,i);
  predPred = predict(:,i);
  
  validLabels=predLabels(predLabels~=0);
  validPredict=predPred(predLabels~=0);
  
  rmse=[rmse sqrt(mean((validLabels-validPredict).^2))];
end