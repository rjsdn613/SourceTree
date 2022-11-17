## Library
import netCDF4 as nc
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from numpy import linspace
from numpy import meshgrid
from mpl_toolkits.axes_grid1 import make_axes_locatable  # divider = make_axes_locatable(gca); LIB
import read_bst_file2 as tc
import read_bst_file as tc2
from collections import Counter
from random import *
from matplotlib import gridspec
import math


"""
각 5년마다 태풍의 총 갯수
"""
TCs_yr = []
for yr in range(1980, 2020):
    letters = str(tc2.get_tc_lat_yr(yr))
    TCs_yr.append(letters.count("TC_count"))



######################### 데이터 입력 ##################################

grid_int = 0.5 # 그리드 간격

start_ln = 100 # 시작 lon
end_ln   = 190 # 끝  lon

start_lt = 0   # 시작 lat
end_lt   = 60  # 끝 lat

start_yr = 1980 # 시작 년도
end_yr   = 2019 # 끝 년도

#######################################################################


yrs      = end_yr - start_yr + 1 
lts= int((end_lt-start_lt)/grid_int)
lns= int((end_ln-start_ln)/grid_int)


for num in range(40,55,5):   # Major TC 기준(num) 을 40 , 45, 50 으로 가정

    TOTAL_MP = np.zeros([lts,lns, yrs])
    yr_idx = -1

    for y in range(start_yr, end_yr+1):
        yr_idx += 1
        TCnum = list(
            filter(lambda x: type(x) == list, tc2.get_tc_lon_yr(y))
        )  #  [i for i in A if isinstance(i, list)]
        NN = np.size(tc.get_tc_lon_yr(y))
        TC = np.empty([NN, 4])
        TC[:, 0] = tc.get_tc_lon_yr(y)
        TC[:, 1] = tc.get_tc_lat_yr(y)
        TC[:, 2] = tc.get_tc_pres_yr(y)
        TC[:, 3] = tc.get_tc_wind_yr(y)


        cc_idx = 0
        for i in range(len(TCnum)):  ## 해당년도 태풍 갯수
            MP = np.zeros([lts, lns, yrs])  # lt, ln, yr
            for c_idx in range(TCnum[i][1]):  ## 같은 태풍 loop
                lt_idx = -1
                for lt in np.arange(start_lt, end_lt, grid_int):
                    lt_idx += 1
                    ln_idx = -1
                    for ln in np.arange(start_ln, end_ln, grid_int):
                        ln_idx += 1
                        if (
                            lt <= TC[cc_idx + c_idx, 1] < lt + grid_int
                            and ln <= TC[cc_idx + c_idx, 0] < ln + grid_int
                        ):
                            if  TC[cc_idx + c_idx, 3] >= num:
        
                                MP[lt_idx-5:lt_idx+5,ln_idx-5:ln_idx+5,yr_idx] = 1
                                # MP[lt_idx,ln_idx,yr_idx] = 1

            cc_idx = cc_idx + TCnum[i][1]
            TOTAL_MP[:, :, yr_idx] = TOTAL_MP[:, :, yr_idx] + MP[:, :, yr_idx]



    sTC=np.zeros(yrs)
    yr_idx=-1
    for y in range(start_yr, end_yr+1):
        yr_idx+=1
        TCnum = list(
            filter(lambda x: type(x) == list, tc2.get_tc_lon_yr(y))
        ) 
        NN = np.size(tc.get_tc_lon_yr(y))
        TC = np.empty([NN, 1])
        TC[:, 0] = tc.get_tc_wind_yr(y)
        cc_idx = 0
        for i in range(len(TCnum)):  ## 해당년도 태풍 갯수
            for c_idx in range(TCnum[i][1]):  ## 같은 태풍 loop
                if TC[cc_idx + c_idx,0]>= num:
                    sTC[yr_idx]+=1
                    break    
            cc_idx = cc_idx + TCnum[i][1]



    #### 10년씩 평균 #####
    myrs=int(yrs/10)
    MP_10year_sum = np.zeros([lts, lns, myrs])
    sTC_10year_sum = np.zeros(myrs)
    for i in range(myrs):
        MP_10year_sum[:, :, i] = np.sum(TOTAL_MP[:, :, i * 10 : i * 10 + 10], axis=2)
        sTC_10year_sum[i]      = np.sum(sTC[i*10:i*10+10])

    #### 0 을 NaN 처리 ####
    MP_10year_sum=np.where(MP_10year_sum==0,np.NaN,MP_10year_sum)




    ############################################################################################
    # plot
    ############################################################################################
    fig = plt.figure(figsize=(15, 12))  ## 4,4 는 inch

    for years in range(myrs):

        ax = fig.add_subplot(2, 2, years + 1)

        ######## MAP BASIC SETTING START ########

        map = Basemap(
            projection="merc", llcrnrlon=100, llcrnrlat=0, urcrnrlon=170, urcrnrlat=45, resolution="h"
        )
        llons, llats = np.meshgrid(np.arange(start_ln, end_ln, grid_int), np.arange(start_lt, end_lt, grid_int))
        x, y = map(llons, llats)

        map.fillcontinents(color="grey", lake_color="aqua")
        map.drawcoastlines()

        # draw lat lon label on map
        map.drawparallels(np.arange(start_lt, end_lt-10, 10), labels=[1, 0, 0, 0])
        map.drawmeridians(np.arange(start_ln, end_ln-10, 20), labels=[0, 0, 0, 1])
        ######## MAP BASIC SETTING END ########




        ######################## DATA PLOT START ########################
        
        ## colorbar levels 정하기 ##
        MP_10year_sum[np.isnan(MP_10year_sum)]=0 # max 값을 찾기위해 nan을 0으로 바꿈
        mx=int(np.max(MP_10year_sum))

        if mx%6 == 0:
            iterval=int(mx/6)
            level=[0,iterval*1,iterval*2,iterval*3,iterval*4,iterval*5,iterval*6]

        elif mx%6 != 0:
            #mx보다 크고 6으로 나눠떨어지는 수 찾기
            while mx%6 != 0:
                mx=mx+1
            iterval=int(mx/6)
            level=[0,iterval*1,iterval*2,iterval*3,iterval*4,iterval*5,iterval*6]

        MP_10year_sum=np.where(MP_10year_sum==0,np.NaN,MP_10year_sum) #다시 0을 NaN으로 바꿈


        map.contourf(x, y, MP_10year_sum[:, :, years], cmap="Reds", levels=level)
        cbar = map.colorbar()
        cbar.set_label("Major  TCs", rotation=90, size=12)

        ######################## DATA PLOT END ############################



        ######## TITLE SETTING START ########
        tcs = str(sum(TCs_yr[years * 10 : years * 10 + 10]))
        stcs=str(int(sTC_10year_sum[years]))
        title1 = str(years * 10 + start_yr)
        title2 = str(years * 10 + start_yr+9)

        plt.title(title1 + "-" + title2 ,fontsize=13,fontweight='bold')
        plt.title( "TCs = " + tcs + "\n" + "Majors = " + stcs   , loc='right',fontsize=10,fontweight='bold')
        
        
        strnum=str(num)
        plt.title( "Major ≥ "+strnum+"m/s" , loc='left',fontsize=10,fontweight='bold')
        
        str_grid_int=str(grid_int)
        plt.suptitle("Decadal Major TC Areas(" + str_grid_int + "°x" + str_grid_int + "°)", fontsize=20)

        ######## TITLE SETTING END ########

    """
    서브플롯 간격 조절
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)
    wspace와hspace는 서브 플롯 사이에 예약 된 공간을 지정합니다. 축 너비와 높이의 비율입니다.
    left,right,top 및bottom 매개 변수는 서브 플롯의 4면 위치를 지정합니다. 그것들은 그림의 너비와 높이의 분수입니다.
    """
    """
    plt.subplots 함수를 써서 서브플롯을 할 경우에는
    figure, axes = plt.subplots(2,2, constrained_layout=True) 이런식으로 서브풀롯 간격을 자동 적절간격 세팅
    """

    fig.tight_layout()  # 자동 서브플롯 간격 안겹치게

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.85, hspace=0.4, wspace=0.4)

    # plt.show()

    plt.savefig("E:/CSL/visual studio/SourceTree/decadal_"+ strnum +".png")

    plt.close()
