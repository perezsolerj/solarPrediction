function [ coords ] = getCoords( locationName )
  %% Searchs for the coordinates of the given locationName to return longitude latitude coordinates
  
  %% New places can be added concatenating their information to database in the format 'locationName', [longitude latitude]
  database=struct('UJI', [-0.06 39.99]);
  databaseNames=fieldnames(database);
  
  if any(strcmp(locationName,databaseNames))
      coords = database.(locationName);
  else
      error('%s is not a recognized parameter name',locationName)
  end
  
end

