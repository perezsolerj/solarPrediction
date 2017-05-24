%%Process copernicus csv imported file to match MSGCPP
%%Guess the imported CSV is a 528480x1 matrix in UJI2016v2 (may be changed
%%in first line containing 1 year (+1day) information (2016-1-1) -> (2017-1-1)

%%Input
fullData=UJI2015;
dataFolder='../data/';
location='UJI'; %%Edit getCoords to add new locations

%Data
nextYear=2016;
initDay=[2015 1 1];
initMeasure=16;


expLocation=strcat(dataFolder,location);
mkdir(expLocation); %%Should not need to create the folder, but just in case...

yearFolder=strcat(expLocation,'/',num2str(initDay(1)));
mkdir(yearFolder);

monthFolder=strcat(yearFolder,'/',num2str(initDay(2)));
mkdir(monthFolder);



while (initDay(1) ~= nextYear)
    %%process data to extract the next day at 15 mins intervals
    %% No es muy entendible pero mola!
    %%data=fullData( (initMeasure + 1):15:(initMeasure + 60*24 -1));
    data=zeros(24*4,1);
    if (initMeasure>14)
        data(1)=sum(fullData(initMeasure-15:initMeasure));
    end
    for i=2:1:24*4
        if (initMeasure>1)
          data(i)=sum(fullData(initMeasure+((i-2)*15):initMeasure+((i-1)*15)));
        end
    end
    
    save(strcat(monthFolder,'/Copernicus_',num2str(initDay(3))),'data');
    
    %%Jump to next day...
    initMeasure=initMeasure+60*24;
    initDay=nextDay(initDay);
    %%Update year and month folders (just in case...)
    yearFolder=strcat(expLocation,'/',num2str(initDay(1)));
    monthFolder=strcat(yearFolder,'/',num2str(initDay(2)));
    mkdir(yearFolder);
    mkdir(monthFolder);
    
    
end
