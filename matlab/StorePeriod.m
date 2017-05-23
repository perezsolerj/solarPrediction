%%% Downloads and saves the desired period of time in a folder

%Configurable Parameters
initDay=[2016 1 1];
endDay=[2017 1 1];
size=[151 151];
location=[-0.06 39.99];
dataFolder='../data/';
reDownload=0

%Data
missingDays=0;

expLocation=strcat(dataFolder,num2str(location(1)),'-',num2str(location(2)));
mkdir(expLocation);

yearFolder=strcat(expLocation,'/',num2str(initDay(1)));
mkdir(yearFolder);

monthFolder=strcat(yearFolder,'/',num2str(initDay(2)));
mkdir(monthFolder);

while(~isequal(initDay,endDay))
  %%Check if file already exists and if parameter REDOWNLOAD is on to decide to skip it or not
  if exist(strcat(monthFolder,'/',num2str(initDay(3)),'.mat'), 'file') == 2 && reDownload==0
    disp(strcat(monthFolder,'/',num2str(initDay(3)),'.mat',' already downloaded'))
  else
    try %% try and catch to ignore download errors and download the maximum possible days without stopping
      data = downloadFunction('location',location,'size',size,'initDate',initDay);
      save(strcat(monthFolder,'/',num2str(initDay(3))),'data');
    catch
      disp(strcat('Error downloading ',monthFolder,'/',num2str(initDay(3))))
      missingDays=missingDays+1;
    end
  end

  initDay=nextDay(initDay);
  %%Update year and month folders (just in case...)
  yearFolder=strcat(expLocation,'/',num2str(initDay(1)));
  monthFolder=strcat(yearFolder,'/',num2str(initDay(2)));
  mkdir(yearFolder);
  mkdir(monthFolder);
    
end

disp(strcat(num2str(missingDays),' could not be downloaded...try again?'))
