function [ newDay ] = nextDay( initDay )
% returns the next day
  monthDays=[31 28 31 30 31 30 31 31 30 31 30 31];


  %%Jump to next day
  initDay(3)=initDay(3)+1;
  %%Check new month or new year
  if (initDay(3)>monthDays(initDay(2)))
      
    if(initDay(2) ~= 2 || mod(initDay(1),4) ~= 0 || initDay(3)>29) %%Check if it is a leap year...
      initDay(3)=1;
      initDay(2)=initDay(2)+1;
            
      if(initDay(2)>12)
        initDay(1)=initDay(1)+1;
        initDay(2)=1;
              
      end
    end
  end
  newDay=initDay;
  
end

