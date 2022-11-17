"""
순서대로 태풍 하나씩 불러오는 코드

"""
#Library
import netCDF4 as nc
import pandas as pd
import numpy as np
from numpy import linspace
from collections import Counter
from random import *


def get_bst_NWP(idx):
    
    with open("E:/CSL/new_bst_all_80.txt", "r") as f:
        TCs=0
        while True:


            line = f.readline()
            if not line:
                break
            TCs += 1 # 전체 태풍중 몇번째 태풍인지 (연도 상관없이)
            TC_info_line = line.split()
            TC_number = int(TC_info_line[1][2:5])
            TC_count_num = int(TC_info_line[2])
            if idx == TCs:
                if int(TC_info_line[1][0:2]) >= 50:
                    yrs_idx = 1900
                elif int(TC_info_line[1][0:2]) <= 50:
                    yrs_idx = 2000
                print("TC number is", TC_number, "in", int(TC_info_line[1][0:2])+yrs_idx )
                print("grade 2 : Tropical Depression (TD)   \
                       grade 3 : Tropical Storm (TS)        \
                       grade 4 : Severe Tropical Storm (STS)\
                       grade 5 : Typhoon (TY)               \
                       grade 6 : Extra-tropical Cyclone (L) \
                       grade 9 : Tropical Cyclone of TS intensity or higher ")
                TC_data = []
                for i in range(TC_count_num):
                    data = f.readline()
                    data = data.split()
                    data[6] = round(int(data[6])*0.514,2) # knots to m/s, 소수점 2자리까지 표현
                    data[3] = round(int(data[3])*0.1,2) # lat
                    data[4] = round(int(data[4])*0.1,2) # lon
                    TC_data.append(data) 
                    tc=pd.DataFrame(TC_data)                 
                    if len(tc.columns) == 7:
                        tc.columns=['date','indicator 002','grade','lat','lon','pres','spd']    
                        tc=tc[['date','grade','lat','lon','pres','spd']]                    
                    elif len(tc.columns) == 11:
                        tc.columns=['date','indicator 002','grade','lat','lon','pres','spd','','','','']    
                        tc=tc[['date','grade','lat','lon','pres','spd']]                         
            else: 
                for i in range(TC_count_num): 
                    f.readline()
    return tc

