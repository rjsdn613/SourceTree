import netCDF4 as nc
import pandas as pd
import numpy as np
from numpy import linspace
from collections import Counter
from random import *
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy import interpolate
import datetime
from matplotlib.lines import Line2D

#TS급이상만 추출(17m/s 이상)
def KMI_data(idx):
    if idx > 9:
        return print("!! 태풍의 갯수 10개 !!")
    else: 
        print("!!! TS급(17m/s) 이상만 출력(3, 6시간 간격 data 혼합) !!!")
    with open("E:/CSL/힌남노랑 경로 유사한 태풍들.txt", "r", encoding='UTF-8') as f:
        TCs=-1
        data = f.readline()

        while True:
            if data == "END\n":
                break
            TCs += 1 # 전체 태풍중 몇번째 태풍인지 (연도 상관없이)
            TC_info_line = data.split('\t')
            TC_number = int(TC_info_line[0])


            data = f.readline()
            TC_info_line = data.split('\t')
            tc = pd.DataFrame([])
            if idx == TCs:
                while len(TC_info_line[0]) != 6 :
                    
                    data_t = data.split('\t')

                    yr = data_t[0][0:4]
                    mo = data_t[0][5:7]
                    dy = data_t[0][8:10]
                    hr = data_t[0][-5::]


                    lat = round(float(data_t[1]),2) 
                    lon = round(float(data_t[2]),2) 
                    pres = round(int(data_t[3]),2) 
                    if data_t[4] == '-':
                        wspd = -999
                    else:
                        wspd = round(int(data_t[4]),2) 

                    tcdata = pd.DataFrame([[yr,mo,dy,hr,lon,lat,pres,wspd]])
                    tc=pd.concat([tc, tcdata])

                    data = f.readline()
                    if data == "END\n":
                        break                                       
                    TC_info_line = data.split('\t')
                tc.columns=['yr','mo','dy','hr','lon','lat','pres','spd']    
                tc.insert(0, 'num', TC_number)      
                tc.insert(0, 'idx', TCs)
                tc.reset_index(drop=True, inplace=True)
                TC_number = int(TC_info_line[0])    
                return tc[tc.spd >= 17].reset_index(drop=True)
            else:
                while len(TC_info_line[0]) != 6 :
                    data = f.readline()
                    if data == "END\n":
                        break                                       
                    TC_info_line = data.split('\t')
                
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def interp_method2(data, length):
    from scipy import interpolate
    import numpy as np
    x = data.index.to_list()
    y = data.values
    f_linear = interpolate.interp1d(x, y, kind='linear')
    xnew = np.linspace(0, x[-1], num=length, endpoint=True)
    y_new_linear = f_linear(xnew)
    data = y_new_linear
    return(data)
'''
각 태풍의 28도 이후
'''



ncfile_pl = nc.Dataset("E:/CSL/ERA5/pressure_level/ERA5_1982.nc", 'r')

#lat(0N~60N)
#lon(120E~180E)
ncfile_root = nc.Dataset("E:/CSL/ERA5/single_level/ERA5_single_level_1982.nc", 'r')
map_lon = ncfile_root.variables["longitude"][:]#241개
map_lat = ncfile_root.variables["latitude"][:]#241개

'''
******************************************************************************************
<single_level variables>
10m u-component of wind, 10m v-component of wind, Mean sea level pressure, Sea surface temperature, Surface pressure, Total precipitation

<pressure_level variables>
Divergence, Geopotential, Potential vorticity, Specific humidity, U-component of wind, V-component of wind, Vertical velocity, Vorticity (relative)
******************************************************************************************

******************************************************************************************
ERA5(ECMWF; European Centre for Medium-Range Weather Forecasts, 영국) 에서 제공하는 SST는
2007년 9월 이전에는 HadISST2, 2007년 9월 이후에는 OSTIA data를 사용한다.

OSTIA(Operational Sea Surface Temperature and Sea Ice Analysis) : 영국 기상청
HadISST2(Hadley Centre Sea Ice and Sea Surface Temperature dataset) : Met Office Hadley Centre 제공, 영국 기상청 
******************************************************************************************
'''
#Jet mean area : 40-50N, 130-150E
# lat[40:81], lon[40:121]

#1년씩 2952개 times
# 7월1일 ~ 10월 31일


total_tcs = 10
'''
202112	오마이스(OMAIS)	2021/08/20 21:00 ~ 2021/08/24 06:00
201918	미탁(MITAG)	2019/09/28 09:00 ~ 2019/10/03 12:00
201825	콩레이(KONG-REY)	2018/09/29 15:00 ~ 2018/10/07 09:00
201819	솔릭(SOULIK)	2018/08/16 09:00 ~ 2018/08/25 03:00
201618	차바(CHABA)	2016/09/28 03:00 ~ 2016/10/06 00:00
201007	곤파스(KOMPASU)	2010/08/29 21:00 ~ 2010/09/03 03:00
200415	메기(MEGI)	2004/08/16 15:00 ~ 2004/08/20 18:00
200314	매미(MAEMI)	2003/09/06 15:00 ~ 2003/09/14 06:00
201905	다나스(DANAS)	2019/07/16 15:00 ~ 2019/07/20 12:00   
201004	뎬무(DIANMU)	2010/08/08 21:00 ~ 2010/08/12 15:00  	
200006	볼라벤(BOLAVEN)	2000/07/24 09:00 ~ 2000/08/02 09:00

'''


mean_jet = []
sjd = int(datetime.datetime(1982, 7, 1).strftime("%j"))
for i in range(total_tcs):
    tc = KMI_data(i)
    mid_idx = tc.lat[tc.lat == find_nearest(tc.lat, 28)].index[0] #28도에 가장 가까운 인덱스

    mid_yr = int(tc.yr.iloc[mid_idx])
    mid_mo = int(tc.mo.iloc[mid_idx])
    mid_dy = int(tc.dy.iloc[mid_idx])
    mid_hr = int(tc.hr.iloc[mid_idx][0:2])

    mjd = int(datetime.datetime(mid_yr, mid_mo, mid_dy).strftime("%j"))
    mid_date = (mjd-sjd)*24 

    ncfil_pl = nc.Dataset("E:/CSL/ERA5/pressure_level/ERA5_"+str(mid_yr)+".nc", 'r') # 250hPa, 500hPa, 850hPa
    if mid_yr == 2022:
        expver = 1
        mean_jet.append(np.mean(np.mean(ncfil_pl.variables['u'][mid_date, expver, 0, 40:81, 40:121],axis=0),axis=0))#태풍이 중위도위치일때 Jet mean area에서 평균 제트강도
    else:
        mean_jet.append(np.mean(np.mean(ncfil_pl.variables['u'][mid_date, 0, 40:81, 40:121],axis=0),axis=0))#태풍이 중위도위치일때 Jet mean area에서 평균 제트강도

#3 4 3
strong_jet_idx = np.argsort(mean_jet)[::-1][0:3]
mod_jet_idx = np.argsort(mean_jet)[::-1][3:7]
weak_jet_idx = np.argsort(mean_jet)[::-1][7:10]



# varname = ['u200', 'v200', 'u850', 'v850', 'vws', 'sst', 'u10', 'v10', 'slp', 'tp']
# for i in range(len(varname)):
#     globals()[varname[i]] = np.empty([total_tcs, 241, 241])


sjd = int(datetime.datetime(1982, 7, 1).strftime("%j"))
for i in range(total_tcs):
    tc = KMI_data(i)
    mid_idx = tc.lat[tc.lat == find_nearest(tc.lat, 28)].index[0] #28도에 가장 가까운 인덱스

    mid_yr = int(tc.yr.iloc[mid_idx])
    mid_mo = int(tc.mo.iloc[mid_idx])
    mid_dy = int(tc.dy.iloc[mid_idx])
    mid_hr = int(tc.hr.iloc[mid_idx][0:2])

    end_yr = int(tc.yr.iloc[-1])
    end_mo = int(tc.mo.iloc[-1])
    end_dy = int(tc.dy.iloc[-1])
    end_hr = int(tc.hr.iloc[-1][0:2])

    mjd = int(datetime.datetime(mid_yr, mid_mo, mid_dy).strftime("%j"))
    ejd = int(datetime.datetime(end_yr, end_mo, end_dy).strftime("%j"))

    mid_date = (mjd-sjd)*24 
    ly_date = (ejd-sjd)*24 

    #time, expver, level, latitude, longitude(2022 only)
    #time, level, latitude, longitude(else)
    #expver[0]=1 ; ERA5
    #expver[1]=5 ; ERA5T

    '''
    2022년 데이터는 ERA5T 자료로 되어있다. 여름데이터가 없을수도.
    지금 받는거 용량 확인해보고 2배면 걍 2022년은 빼고하자.
    '''

    ncfile_pl = nc.Dataset("E:/CSL/ERA5/pressure_level/ERA5_"+str(mid_yr)+".nc", 'r') # 250hPa, 500hPa, 850hPa
    if mid_yr == 2022:
        expver = 1
        u200 = ncfile_pl.variables['u'][mid_date:ly_date+1, expver, 0, :, :] 
        v200 = ncfile_pl.variables['v'][mid_date:ly_date+1, expver, 0, :, :] 
        u850 = ncfile_pl.variables['u'][mid_date:ly_date+1, expver, 2, :, :] 
        v850 = ncfile_pl.variables['v'][mid_date:ly_date+1, expver, 2, :, :] 
    else:
        u200 = ncfile_pl.variables['u'][mid_date:ly_date+1, 0, :, :] 
        v200 = ncfile_pl.variables['v'][mid_date:ly_date+1, 0, :, :] 
        u850 = ncfile_pl.variables['u'][mid_date:ly_date+1, 2, :, :] 
        v850 = ncfile_pl.variables['v'][mid_date:ly_date+1, 2, :, :] 

    vws = ((u200-u850)**2 + (v200-v850)**2)**0.5



    ncfil_sl = nc.Dataset("E:/CSL/ERA5/single_level/ERA5_single_level_"+str(mid_yr)+".nc", 'r')
    if mid_yr == 2022:
        expver = 1
        u10 = ncfil_sl.variables['u10'][mid_date:ly_date+1, expver, :, :]
        v10 = ncfil_sl.variables['v10'][mid_date:ly_date+1, expver, :, :]
        slp = ncfil_sl.variables['msl'][mid_date:ly_date+1, expver, :, :]
        sst = ncfil_sl.variables['sst'][mid_date:ly_date+1, expver, :, :]
        tp = ncfil_sl.variables['tp'][mid_date:ly_date+1, expver, :, :] #Total precipitation
    else:
        u10 = ncfil_sl.variables['u10'][mid_date:ly_date+1, :, :]
        v10 = ncfil_sl.variables['v10'][mid_date:ly_date+1, :, :]
        slp = ncfil_sl.variables['msl'][mid_date:ly_date+1, :, :]
        sst = ncfil_sl.variables['sst'][mid_date:ly_date+1, :, :]
        tp = ncfil_sl.variables['tp'][mid_date:ly_date+1, :, :] #Total precipitation










    varname = ['u200', 'v200', 'u850', 'v850', 'vws', 'sst', 'u10', 'v10', 'slp', 'tp']
    for k in range(len(varname)):
        globals()[varname[k]+'_'+str(i)] = eval(varname[k])











'''
#1시간별로 태풍위치를 계속 추적할때, slp 최저지점을 태풍위치로 봐도 무방한지 테스트해야한다.
#베스트트랙과 slp 최저지점 트랙킹한것이 비슷한지 확인하기
#slp에서 베스트트랙상 태풍위치 반경0.5도 이내에서 slp 최저점 찾아서 찍기
'''
#KMI_data(6)은 ERA5에서 slp 데이터가 없다.
#KMI_data(0)은 ERA5 데이터가 missing 값이 많아 쓸 수 없다.
def TC_loc_verification(i):
    tc = KMI_data(i)

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    map = Basemap(projection='mill',llcrnrlat=0,urcrnrlat=50,llcrnrlon=100,urcrnrlon=180)
    map.drawcoastlines()
    map.drawmeridians(np.arange(0, 360, 30), labels=[0,0,0,1])
    map.drawparallels(np.arange(-90, 90, 10), labels=[1,0,0,0])

    for j in range(len(tc)):
        tcyr = int(tc.yr.iloc[j])
        tcmo = int(tc.mo.iloc[j])
        tcdy = int(tc.dy.iloc[j])
        tclon = tc.lon.iloc[j]
        tclat = tc.lat.iloc[j]
        tcloc_idx = (int(datetime.datetime(tcyr, tcmo, tcdy).strftime("%j")) - sjd)*24 + int(tc.hr.iloc[j][0:2])

        lon_idx = np.where(map_lon==find_nearest(map_lon, tclon))[0][0]
        lat_idx = np.where(map_lat==find_nearest(map_lat, tclat))[0][0]

        ncfil_sl = nc.Dataset("E:/CSL/ERA5/single_level/ERA5_single_level_"+str(tcyr)+".nc", 'r')
        if tcyr == 2022:
            expver = 1
            slp = ncfil_sl.variables['msl'][tcloc_idx, expver, lat_idx-12:lat_idx+13, lon_idx-12:lon_idx+13]
            #추정최저기압
            mslp = np.min(slp)*0.01 #Pa -> hPa
            p_tc_pres = np.round(mslp, 2) #Presumed tc pres

            #추정최저기압의 위치
            lt_idx, ln_idx = np.where(slp==np.min(slp))
            p_tc_lon, p_tc_lat = map_lon[ln_idx[0]+lon_idx], map_lat[lt_idx[0]+lat_idx] #전체격자 기준이므로 lon_idx, lat_idx 더함

        else:
            slp = ncfil_sl.variables['msl'][tcloc_idx, lat_idx-1:lat_idx+2, lon_idx-1:lon_idx+2]

            #추정최저기압
            mslp = np.min(slp)*0.01 #Pa -> hPa
            p_tc_pres = np.round(mslp, 2) #Presumed tc pres

            #추정최저기압의 위치
            lt_idx, ln_idx = np.where(slp==np.min(slp))
            p_tc_lon, p_tc_lat = map_lon[ln_idx[0]+lon_idx], map_lat[lt_idx[0]+lat_idx] #전체격자 기준이므로 lon_idx, lat_idx 더함

        p_tclon,p_tclat = map(p_tc_lon,p_tc_lat) #ERA5 추정값
        b_tclon,b_tclat = map(tclon,tclat) #베스트트랙 값

        plt.plot(p_tclon,p_tclat, '-bo', markersize=4, label="ERA5(mslp tracking) TC track") 
        plt.plot(b_tclon,b_tclat, '-go', markersize=4, label="KMI-bst TC track") 


    #한시간 간격으로 주변 최저 mslp 찍어도 트랙이 잘 찍히나?
    start_idx = (int(datetime.datetime(int(tc.yr.iloc[0]), int(tc.mo.iloc[0]), int(tc.dy.iloc[0])).strftime("%j")) - sjd)*24 + int(tc.hr.iloc[0][0:2])
    end_idx = (int(datetime.datetime(int(tc.yr.iloc[-1]), int(tc.mo.iloc[-1]), int(tc.dy.iloc[-1])).strftime("%j")) - sjd)*24 + int(tc.hr.iloc[-1][0:2])

    #best-track 자료를 1시간 간격으로 interpolate 한 다음, 내삽된 위치를 중심으로 반경N도 이내에 최저 mslp 지점을 찾아 찍는다.
    #이것이 베스트 트랙자료와 비슷하면 합리적으로 mslp가 태풍위치로 추정된 것
    interp_tclat = interp_method2(tc.lat, end_idx-start_idx+1)
    interp_tclon = interp_method2(tc.lon, end_idx-start_idx+1)

    for k in range(len(interp_tclat)):  
        lon_idx = np.where(map_lon==find_nearest(map_lon, interp_tclon[k]))[0][0]
        lat_idx = np.where(map_lat==find_nearest(map_lat, interp_tclat[k]))[0][0]

        slp = ncfil_sl.variables['msl'][start_idx + k , lat_idx-2:lat_idx+3, lon_idx-2:lon_idx+3] # 0.5도 반경이내에서 최저mslp 찾아서 찍기
        lt_idx, ln_idx = np.where(slp==np.min(slp))
        phr_tc_lon, phr_tc_lat = map_lon[ln_idx[0]+lon_idx], map_lat[lt_idx[0]+lat_idx] 
        phr_tclon,phr_tclat = map(phr_tc_lon,phr_tc_lat) #ERA5 시간별 추정값
        plt.plot(phr_tclon,phr_tclat, '-ro', markersize=4, label="ERA5 hourly TC track") 

    #custom legend
    custom_lines = [Line2D([0], [0], color='b', lw=4),
                    Line2D([0], [0], color='g', lw=4),
                    Line2D([0], [0], color='r', lw=4)]
    plt.legend(custom_lines, ['ERA5(mslp tracking) TC track', 'KMI-bst TC track', 'ERA5 hourly TC track'])

