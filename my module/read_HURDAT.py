'''
Each storm entry begins with a header line, listing the storm ATCF code, the storm name, and the number of fixes to follow. 
There follows a line for each storm fix where the values are comma delimited. 
The first entry is the date of the fix in year, month, day. 
Next is the time in UTC, then a letter code for special entries:
L - Landfall (the wind center crosses a major coastline)
W - Maximum sustained wind speed for storm history
P - Minimum central pressure for storm history
I - Intensity peak for storm history in terms of both pressure and wind
C - Closest approach to coast not followed by a landfall
S - Status change in the system
G - Genesis of the system
T - An additional fix point added to clarify track detail

Each fix is generally 6 hours from the last except when a special fix is noted, such as landfall.
The storm status follows given by a two-letter code:

WV - Tropical Wave
TD - Tropical Depression
TS - Tropical Storm
HU - Hurricane (winds > 64 kt)
EX - Extratropical cyclone
SD - Subtropical depression (winds <34 kt)
SS - Subtropical storm (winds >34 kt)
LO - A low pressure system not fitting any of above descriptions
DB - non-tropical Disturbance not have a closed circulation

Next is the latitude and longitude of the fix with a letter code giving the hemisphere. 

Next is a series of numbers, the first is the maximum sustained winds at this time in knots.

The second number is the minimum central pressure at the time in milibars.

The rest of the numbers represent the radii in nautical miles of certain wind speeds in the Northeast, Southeast, Southwest, and Northwest quadrants in that order. 
The first four numbers are the radii for the 34 knot winds, the next four are for the 50 knot winds, and the last for the 64 knot (hurricane force) winds.
Time is given in Universal Time Code (UTC) which is a 24 hour clock rendering of 
the the time at 0 degrees Longitude or Greenwich Mean Time (to convert to Eastern Standard Time add five hours, to convert to Eastern Daylight Savings Time add four hours). 
Latitude and Longitude are to the nearest tenth of a degree and the Hemispheres are listed as 'N' or 'S' for North or South and 'E' or 'W' for East or West. 
Wind direction is degrees clockwise from due north (0 deg.).
Wind speed is given in nautical miles per hour. (=kn, 1시간에 1해리(약 1852m) 가는 속도); (약 0.514 m/s)
The central minimum pressure is given in millibars.(1mb = 1hPa)
'''
import netCDF4 as nc
import pandas as pd
import numpy as np
from numpy import linspace
   

def ATL(idx):
        
    with open("E:/CSL/Atlantic hurricane database (HURDAT2) 1851-2020.txt", "r") as f:
        TCs=0
        while True:


            line = f.readline()
            if not line:
                break
            TCs += 1 # 전체 태풍중 몇번째 태풍인지 (연도 상관없이)
            TC_info_line = line.split(',')
            TC_number = int(TC_info_line[0][2:4])
            TC_count_num = int(TC_info_line[2])

            if idx == TCs:
                # print("TC number is", TC_number, "in", int(TC_info_line[0][4:8]))
                # print("TC number (1294/ 1924) is start of 1981 year/ end of 2020 ")
                TC_data = []
                for i in range(TC_count_num):
                    data = f.readline()
                    data = data.split(',')
                    data[6] = round(int(data[6])*0.514,2) # knots to m/s, 소수점 2자리까지 표현
                    TC_data.append(data) 
                    tc=pd.DataFrame(TC_data) 
                    tc.columns=['date','UTC','special entries','grade','lat','lon','spd','pres',\
                        '','','','','','','','','','','','','']  
                    tc=tc[['date','UTC','grade','lat','lon','spd','pres']]
            else: 
                for i in range(TC_count_num): 
                    f.readline()
    return tc





def NEP(idx):
    
    with open("E:/CSL/Northeast and North Central Pacific hurricane database (HURDAT2) 1949-2020.txt", "r") as f:
        TCs=0
        while True:


            line = f.readline()
            if not line:
                break
            TCs += 1 # 전체 태풍중 몇번째 태풍인지 (연도 상관없이)
            TC_info_line = line.split(',')
            TC_number = int(TC_info_line[0][2:4])
            TC_count_num = int(TC_info_line[2])

            if idx == TCs:
                # print("TC number is", TC_number, "in", int(TC_info_line[0][4:8]))
                # print("TC number (374/ 1169) is start of 1981 year/ end of 2020 ")
                TC_data = []
                for i in range(TC_count_num):
                    data = f.readline()
                    data = data.split(',')
                    data[6] = round(int(data[6])*0.514,2) # knots to m/s, 소수점 2자리까지 표현
                    TC_data.append(data) 
                    tc=pd.DataFrame(TC_data) 
                    tc.columns=['date','UTC','special entries','grade','lat','lon','spd','pres',\
                        '','','','','','','','','','','','','']  
                    tc=tc[['date','UTC','grade','lat','lon','spd','pres']]
            else: 
                for i in range(TC_count_num): 
                    f.readline()
    return tc

