%%% Calculates the results assuming labels and predict variables are
%%% created and have the shape [instances, predictionInstants]

months=8;

rmse=[];
bias=[];
sde=[];

for month=1:months
  rmseM=[];
  biasM=[];
  sdeM=[];
  
  pointsMonth=floor(size(labels,1)/months);
  
  labelsM=labels( ((month-1)*pointsMonth+1):(month*pointsMonth));
  predictM=predict( ((month-1)*pointsMonth+1):(month*pointsMonth));
  
  for i=1:size(labels,2)
    
    [rmse_ sde_ bias_] = computeErrors(labelsM, predictM);
  
    rmseM = [rmseM rmse_];
    biasM = [biasM bias_];
    sdeM  = [sdeM sde_];
  end
  
  rmse= [rmse rmseM];
  bias= [bias biasM];
  sde = [sde sdeM];
end