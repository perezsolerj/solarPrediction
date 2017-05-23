function y = downloadFunction( varargin )
%%Downloads the desired location and days information

  nArgs = length(varargin);
  
  if round(nArgs/2)~=nArgs/2
    error('EXAMPLE needs propertyName/propertyValue pairs')
  end
  
  options = struct('location',[-0.06 39.99], 'size', [1 1], 'initdate', [2016 1 8]);
  optionNames = fieldnames(options);
  
  for pair = reshape(varargin,2,[]) %# pair is {propName;propValue}
    inpName = lower(pair{1}); %# make case insensitive
    if any(strcmp(inpName,optionNames))
      %# overwrite options. If you want you can test for the right class here
      %# Also, if you find out that there is an option you keep getting wrong,
      %# you can use "if strcmp(inpName,'problemOption'),testMore,end"-statements
      options.(inpName) = pair{2};
    else
      error('%s is not a recognized parameter name',inpName)
    end
  end
  
  if mod(options.('size')(1),2)~=1 || mod(options.('size')(2),2)~=1
      error('size parameter needs to be an odd number so the location can be centered');
  end
  
  resolution=0.03;
  initDate= options.('initdate');
  size=options.('size');
  
  minlocation = options.('location') - (size - 1)/ 2 .* resolution;
  maxlocation = options.('location') + (size - 1)/ 2 .* resolution;

  urlwrongmax=15;
  failvalue=-100;

  
  str1='http://msgcpp-ogc-archive.knmi.nl/msgar.cgi? &service=wcs&version=1.0.0&request=getcoverage&coverage=surface_downwelling_shortwave_flux_in_air&FORMAT=AAIGRID&CRS=EPSG%3A4326&';
 
% %Longitude and latitude  STR2 --> BBOX=-0.06,39.99,-0.06,39.99&
 
  str2=strcat('BBOX=',num2str(minlocation(1)),',',num2str(minlocation(2)),',',num2str(maxlocation(1)),',',num2str(maxlocation(2)),'&');

% %Width and height of the image  STR3 --> WIDTH=1&HEIGHT=1&
  
 str3=strcat('WIDTH=',num2str(size(1)),'&HEIGHT=',num2str(size(2)),'&');
 
% %Date STR4 --> time=2016-08-25T

  str4=strcat('time=',num2str(initDate(1)),'-',num2str(initDate(2)),'-',num2str(initDate(3)),'T');
  
% %Time  STR5 -->12:45:00Z

  warning('off','MATLAB:urlread:ReplacingSpaces')

  values=[];
  position=1;
  
  disp(strcat('Downloading->',num2str(initDate(3)), '/', num2str(initDate(2)), '/', num2str(initDate(1))))
  for j=0:1:23 %%Hours
    for i=0:15:59  %%Minutes
        hour=num2str(j);
        minute=num2str(i);
        
        str5=strcat(hour,':',minute,':00Z');
        address=strcat(str1,str2,str3,str4,str5);   
        
        web=urlread(address,'Timeout',15);
        data=strsplit(web);       
        
        urlwrong=0;
        err=0;
        while (strcmp(data{1},'Invalid')&& err==0) %Try to download data again if something fails
            web=urlread(address);
            data=strsplit(web); 
            urlwrong=urlwrong+1;
            if urlwrong==urlwrongmax
                urlwrongmax
                strcat(str5,' ERROR')
                err=1;
            end
        end
       
        if err==0 % Data correctly downloaded
            nodatapos=1;
            while ~strcmp(data{nodatapos},'NODATA_value')
                nodatapos=nodatapos+1;
            end
            nodatapos=nodatapos+1;% Position of the nodata_value number
            
            NODATA_value=str2num(data{nodatapos});
            rows=str2num(data{4});
            columns=str2num(data{2});
        
        
            matrix=zeros(rows,columns);

            counter=1;
            for r=1:rows
                for c=1:columns
                    matrix(r,c)=str2num(data{nodatapos+counter});
                    if matrix(r,c)==NODATA_value
                        matrix(r,c)=0;
                    end
                    counter=counter+1;
                end
            end
            
        elseif err==1 %If data can't be downloaded, fills the matrix with the failvalue
            rows=str2num(size(1));
            columns=str2num(size(2));
            
            for r=1:rows
                for c=1:columns
                    matrix(r,c)=failvalue;
                end
            end
            
        end
   
        values(:,:,position)=matrix;
        position=position+1;        
    end
  end
  
  y=values;
end

