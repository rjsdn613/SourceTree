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
import itertools
import numbers
import math
from obspy.geodetics import kilometers2degrees
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from celluloid import Camera

#v1 data에서
#덴빈(6)은 ERA5에서 slp 데이터가 없다.
#힌남노(0)는 ERA5 데이터가 missing 값이 많아 쓸 수 없다.
#볼라벤(12)은 이동속도 데이터가 없다

#TS급이상만 추출(17m/s 이상)
def KMI_data(idx):
    if idx > 9:
        return print("!! 태풍의 갯수 10개 !!")
    else: 
        print("!!! TS급(17m/s) 이상만 출력(3, 6시간 간격 data 혼합) !!!")
    with open("E:/CSL/힌남노랑 경로 유사한 태풍들_v3.txt", "r", encoding='UTF-8') as f:
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
                    
                    trans_spd = int(data_t[10]) #km/h

                    tcdata = pd.DataFrame([[yr,mo,dy,hr,lon,lat,pres,wspd,trans_spd]])
                    tc=pd.concat([tc, tcdata])

                    data = f.readline()
                    if data == "END\n":
                        break                                       
                    TC_info_line = data.split('\t')

                tc.columns=['yr','mo','dy','hr','lon','lat','pres','spd','trans_spd']    
                tc.insert(0, 'num', TC_number)      
                tc.insert(0, 'idx', TCs)
                tc.reset_index(drop=True, inplace=True)
                if data != "END\n":   
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

def interp_method1(data, length):
    from scipy import interpolate
    import numpy as np
    x = np.arange(len(data))
    y = data
    f_linear = interpolate.interp1d(x, y, kind='linear')
    xnew = np.linspace(0, x[-1], num=length, endpoint=True)
    y_new_linear = f_linear(xnew)
    data = y_new_linear
    return(data)

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

def interp_method3(data, length):
    from scipy import interpolate
    import numpy as np
    x = data.reset_index().index.to_list()
    y = data.values
    f_linear = interpolate.interp1d(x, y, kind='linear')
    xnew = np.linspace(0, x[-1], num=length, endpoint=True)
    y_new_linear = f_linear(xnew)
    data = y_new_linear
    return(data)

class GeoUtil:
    def degree2radius(degree):
        return degree * (math.pi/180)

    def get_harversine_distance(x1, y1, x2, y2, round_decimal_digits=2):
        '''
        경위도 (x1, y1)과 (x2, y2), 두 점 사이의 거리를 반환
        Harversine Formula를 이용하여 2개의 경위도간 거리를 구함(단위:km)
        '''
        if x1 is None or y1 is None or x2 is None or y2 is None:
            return None
        assert isinstance(x1, numbers.Number) and -180 <= x1 and x1 <= 180
        assert isinstance(y1, numbers.Number) and  -90 <= y1 and y1 <=  90
        assert isinstance(x2, numbers.Number) and -180 <= x2 and x2 <= 180
        assert isinstance(y2, numbers.Number) and  -90 <= y2 and y2 <=  90

        R = 6371 # 지구의 반경(단위: km)
        dLon = GeoUtil.degree2radius(x2-x1)    
        dLat = GeoUtil.degree2radius(y2-y1)

        a = math.sin(dLat/2) * math.sin(dLat/2) \
            + (math.cos(GeoUtil.degree2radius(y1)) \
              *math.cos(GeoUtil.degree2radius(y2)) \
              *math.sin(dLon/2) * math.sin(dLon/2))
        b = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return round(kilometers2degrees(R * b), round_decimal_digits)




'''
#1시간별로 태풍위치를 계속 추적할때, slp 최저지점을 태풍위치로 봐도 무방한지 테스트해야한다.
#베스트트랙과 slp 최저지점 트랙킹한것이 비슷한지 확인하기
#slp에서 베스트트랙상 태풍위치 반경0.5도 이내에서 slp 최저점 찾아서 찍기
'''

def TC_loc_verification(i):
    ncfile_root = nc.Dataset("E:/CSL/ERA5/single_level/ERA5_single_level_1982.nc", 'r')
    map_lon = ncfile_root.variables["longitude"][:]#241개
    map_lat = ncfile_root.variables["latitude"][:]#241개

    tc = KMI_data(i)

    fig, ax = plt.subplots(1,1, figsize=(10,10))
    map = Basemap(projection='mill',llcrnrlat=0,urcrnrlat=50,llcrnrlon=120,urcrnrlon=150)
    map.drawcoastlines()
    map.drawmeridians(np.arange(0, 360, 30), labels=[0,0,0,1])
    map.drawparallels(np.arange(-90, 90, 10), labels=[1,0,0,0])

    for j in range(len(tc)):
        tcyr = int(tc.yr.iloc[j])
        tcmo = int(tc.mo.iloc[j])
        tcdy = int(tc.dy.iloc[j])
        tclon = tc.lon.iloc[j]
        tclat = tc.lat.iloc[j]
        tcloc_idx = (int(datetime.datetime(tcyr, tcmo, tcdy).strftime("%j")) - sjd -1)*24 + int(tc.hr.iloc[j][0:2])

        lon_idx = np.where(map_lon==find_nearest(map_lon, tclon))[0][0]
        lat_idx = np.where(map_lat==find_nearest(map_lat, tclat))[0][0]

        ncfil_sl = nc.Dataset("E:/CSL/ERA5/single_level/ERA5_single_level_"+str(tcyr)+".nc", 'r')

        steps = map_lon[2] - map_lon[1] #격자 간격
        around_degree = 2         #주변 몇도만 탐색?
        step_idx = int(around_degree/steps) # 몇도 탐색을 위한 인덱스 개수

        slp = ncfil_sl.variables['msl'][tcloc_idx, lat_idx-around_degree:lat_idx+around_degree+1, lon_idx-around_degree:lon_idx+around_degree+1]

        #추정최저기압
        mslp = np.min(slp)*0.01 #Pa -> hPa
        p_tc_pres = np.round(mslp, 2) #Presumed tc pres

        #추정최저기압의 위치
        lt_idx, ln_idx = np.where(slp==np.min(slp))
        p_tc_lon, p_tc_lat = map_lon[ln_idx[0]+lon_idx-around_degree], map_lat[lt_idx[0]+lat_idx-around_degree] #전체격자 기준이므로 lon_idx, lat_idx 더함

        p_tclon,p_tclat = map(p_tc_lon,p_tc_lat) #ERA5 추정값
        b_tclon,b_tclat = map(tclon,tclat) #베스트트랙 값

        plt.plot(p_tclon,p_tclat, '-bo', markersize=4, label="ERA5(mslp tracking) TC track") 
        plt.plot(b_tclon,b_tclat, '-go', markersize=4, label="KMI-bst TC track") 


    #custom legend
    custom_lines = [Line2D([0], [0], color='b', lw=4),
                    Line2D([0], [0], color='g', lw=4)
                    ]
    plt.legend(custom_lines, ['ERA5(mslp tracking) TC track', 'KMI-bst TC track'])




















total_tcs = 10








#lat(0N~60N)
#lon(120E~180E)
ncfile_root = nc.Dataset("E:/CSL/ERA5/single_level/ERA5_single_level_1982.nc", 'r')
map_lon = ncfile_root.variables["longitude"][:]#241개
map_lat = ncfile_root.variables["latitude"][:]#241개

#Jet mean area : 40-50N, 130-150E
# lat[40:81], lon[40:121]

#1년씩 2952개 times
# 7월1일 ~ 10월 31일

'''
각 태풍의 28도 이후
'''

#제트기류 강도 측정
mean_jet = []
alpha_jd = int(datetime.datetime(1982, 7, 1).strftime("%j"))
for i in range(total_tcs):
    tc = KMI_data(i)
    mid_idx = tc.lat[tc.lat == find_nearest(tc.lat, 28)].index[0] #28도에 가장 가까운 인덱스

    mid_yr = int(tc.yr.iloc[mid_idx])
    mid_mo = int(tc.mo.iloc[mid_idx])
    mid_dy = int(tc.dy.iloc[mid_idx])
    mid_hr = int(tc.hr.iloc[mid_idx][0:2])

    mjd = int(datetime.datetime(mid_yr, mid_mo, mid_dy).strftime("%j"))
    mid_date = (mjd-alpha_jd)*24 

    ncfil_pl = nc.Dataset("E:/CSL/ERA5/pressure_level/ERA5_"+str(mid_yr)+".nc", 'r') # 250hPa, 500hPa, 850hPa
    mean_jet.append(np.mean(np.mean(ncfil_pl.variables['u'][mid_date, 0, 40:81, 40:121],axis=0),axis=0))#태풍이 중위도위치일때 Jet mean area에서 평균 제트강도

#3 4 3
strong_jet_idx = np.argsort(mean_jet)[::-1][0:3]
mod_jet_idx = np.argsort(mean_jet)[::-1][3:7]
weak_jet_idx = np.argsort(mean_jet)[::-1][7:10]



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

'''
# #land-maks load, 나중에 태풍 상륙 체크하기위함 , 한반도육지+3도안에 태풍중심이 들어오면 영향권진입으로 보고 ->  영향권 진입~이탈까지의 강도변화
# lsmask_nc = nc.Dataset("E:/CSL/SST/lsmask.oisst.v2.nc", 'r')
# lsmask_lat= lsmask_nc.variables['lat'][359:600]
# lsmask_lon = lsmask_nc.variables['lon'][479:720]
# lsmask = lsmask_nc.variables['lsmask'][0,359:600,479:720]  #lon[479:720], lat[359,600] / 241개*241개

'''
fig, ax = plt.subplots(1,1,figsize=(15,15))
map = Basemap(projection='mill',
llcrnrlat=0, 
urcrnrlat=50,
llcrnrlon=120, 
urcrnrlon=150 )

map.drawcoastlines()
map.drawmeridians(np.arange(0, 360, 2), labels=[0,0,0,1])
map.drawparallels(np.arange(-90, 90, 2), labels=[1,0,0,0])
x,y=map(lsmask_lon,lsmask_lat)
plt.contourf(x,y,lsmask)

<대한민국>
위도:34~38도
경도:126~130
지역에서만 lsmask 확장

np.where(lsmask_lat==find_nearest(lsmask_lat,34))
np.where(lsmask_lat==find_nearest(lsmask_lat,36))

np.where(lsmask_lon==find_nearest(lsmask_lon,126))
np.where(lsmask_lon==find_nearest(lsmask_lon,130))
'''
# kor_lat_down = np.where(lsmask_lat==find_nearest(lsmask_lat,34))[0][0]
# kor_lat_up =np.where(lsmask_lat==find_nearest(lsmask_lat,38))[0][0]+1

# kor_lon_left = np.where(lsmask_lon==find_nearest(lsmask_lon,126))[0][0]
# kor_lon_right = np.where(lsmask_lon==find_nearest(lsmask_lon,130))[0][0]+1

# ex_lsmask = lsmask.copy()
# ori_lsmask = lsmask.copy()
# degree = 3
# interv = 0.25
# steps = int(degree/interv)
# for i,j in itertools.product(range(kor_lat_down, kor_lat_up),range(kor_lon_left, kor_lon_right)):
#     if lsmask[i,j] == 0:
#         ori_lsmask[i,j] = 999
#         ex_lsmask[i-steps:i, j:j+steps]=-1 ; ex_lsmask[i, j:j+steps]=-1 ; ex_lsmask[i:i+steps, j:j+steps]=-1
#         ex_lsmask[i-steps:i, j]=-1      ; ex_lsmask[i, j]=-1      ; ex_lsmask[i:i+steps, j]=-1
#         ex_lsmask[i-steps:i, j-steps:j]=-1 ; ex_lsmask[i, j-steps:j]=-1 ; ex_lsmask[i:i+steps, j-steps:j]=-1


# #lsmask plot
# fig, [ax1,ax2] = plt.subplots(1,2,figsize=(10,5))
# map = Basemap(projection='mill',
# llcrnrlat=30, 
# urcrnrlat=45,
# llcrnrlon=120, 
# urcrnrlon=137,
# ax=ax1 )

# map.drawcoastlines()
# map.drawmeridians(np.arange(0, 360, 5), labels=[0,0,0,1])
# map.drawparallels(np.arange(-90, 90, 5), labels=[1,0,0,0])
# x,y=map(lsmask_lon,lsmask_lat)
# p1 = ax1.contourf(x,y,ori_lsmask)

# map = Basemap(projection='mill',
# llcrnrlat=30, 
# urcrnrlat=45,
# llcrnrlon=120, 
# urcrnrlon=137,
# ax=ax2 )

# map.drawcoastlines()
# map.drawmeridians(np.arange(0, 360, 5), labels=[0,0,0,1])
# map.drawparallels(np.arange(-90, 90, 5), labels=[1,0,0,0])
# x,y=map(lsmask_lon,lsmask_lat)
# p2 = ax2.contourf(x,y,ex_lsmask)
# plt.tight_layout()
























#베스트트랙과 맞춰서 slp최저점(중심) 반경3도 평균 wspd10으로 나타내기
#처음 불러올때부터 1시간 간격으로 변수를 불러와야돼
varnames = ['mslp', 'tp', 'wspd10','wspd100','wspd850']
alpha_jd = int(datetime.datetime(1982, 7, 1).strftime("%j"))
for i in range(total_tcs):
    tc = KMI_data(i)

    mid_idx = tc.lat[tc.lat == find_nearest(tc.lat, 28)].index[0] #28도에 가장 가까운 인덱스

    start_yr = int(tc.yr.iloc[0])
    start_mo = int(tc.mo.iloc[0])
    start_dy = int(tc.dy.iloc[0])
    start_hr = int(tc.hr.iloc[0][0:2])

    mid_yr = int(tc.yr.iloc[mid_idx])
    mid_mo = int(tc.mo.iloc[mid_idx])
    mid_dy = int(tc.dy.iloc[mid_idx])
    mid_hr = int(tc.hr.iloc[mid_idx][0:2])

    end_yr = int(tc.yr.iloc[-1])
    end_mo = int(tc.mo.iloc[-1])
    end_dy = int(tc.dy.iloc[-1])
    end_hr = int(tc.hr.iloc[-1][0:2])

    sjd = int(datetime.datetime(start_yr, start_mo, start_dy).strftime("%j"))
    mjd = int(datetime.datetime(mid_yr, mid_mo, mid_dy).strftime("%j"))
    ejd = int(datetime.datetime(end_yr, end_mo, end_dy).strftime("%j"))


    start_date = ((sjd-alpha_jd)*24) + start_hr
    mid_date = ((mjd-alpha_jd)*24) + mid_hr
    ly_date = ((ejd-alpha_jd)*24)  + end_hr

    if i == 4 or i == 6:
        start_date = start_date -36
        ly_date = ly_date -36
        
    #Pressure level data(time, level, latitude, longitude)
    ncfile_pl = nc.Dataset("E:/CSL/ERA5/pressure_level/ERA5_"+str(start_yr)+".nc", 'r') # 250hPa, 500hPa, 850hPa
    # u200 = ncfile_pl.variables['u'][start_date:ly_date+1, 0, :, :] 
    # v200 = ncfile_pl.variables['v'][start_date:ly_date+1, 0, :, :] 
    # div200 = ncfile_pl.variables['d'][start_date:ly_date+1, 0, :, :] 

    u850 = ncfile_pl.variables['u'][start_date:ly_date+1, 2, :, :] 
    v850 = ncfile_pl.variables['v'][start_date:ly_date+1, 2, :, :] 
    # q850 = ncfile_pl.variables['q'][start_date:ly_date+1, 2, :, :] #Specific humidity (kg kg-1)
    # vo850 = ncfile_pl.variables['vo'][start_date:ly_date+1, 2, :, :] #Relative vorticity (s-1)
    # w500 = ncfile_pl.variables['w'][start_date:ly_date+1, 1, :, :] #Vertical velocity(Pa s-1)

    # vws = ((u200-u850)**2 + (v200-v850)**2)**0.5
    wspd850 = (u850**2 + v850**2)**0.5


    #Single level data
    ncfil_sl = nc.Dataset("E:/CSL/ERA5/single_level/ERA5_single_level_"+str(start_yr)+".nc", 'r')
    ncfil_sl2 = nc.Dataset("E:/CSL/ERA5/single_level2/ERA5_single_level2_"+str(start_yr)+".nc", 'r')

    u10 = ncfil_sl.variables['u10'][start_date:ly_date+1, :, :]
    v10 = ncfil_sl.variables['v10'][start_date:ly_date+1, :, :]
    u100 = ncfil_sl2.variables['u100'][start_date:ly_date+1, :, :]
    v100 = ncfil_sl2.variables['v100'][start_date:ly_date+1, :, :]
    mslp = ncfil_sl.variables['msl'][start_date:ly_date+1, :, :]*0.01 #Pa -> hPa
    # sst = ncfil_sl.variables['sst'][start_date:ly_date+1, :, :]-273.15 # K -> Celsius
    tp = ncfil_sl.variables['tp'][start_date:ly_date+1, :, :]*1000  #Total precipitation (m) -> mm
    '''
    tp : 물이 그리드 상자에 고르게 퍼졌을 때의 깊이입니다. 
    모델 매개변수를 관찰과 비교할 때 주의를 기울여야 합니다. 
    관찰은 종종 모델 격자 상자에 대한 평균을 나타내기보다는 공간과 시간의 특정 지점에 국한되기 때문입니다.
    '''

    wspd10 = (u10**2 + v10**2)**0.5
    wspd100 = (u100**2 + v100**2)**0.5


    for p in range(len(varnames)):
        globals()[varnames[p]+'_'+str(i)] = eval(varnames[p])


    #한번에 interp하니까 잘 안맞다. 관측기록있는 지점마다 끊어서 interpolate 해야할 듯.
    interp_tclat1 = []
    interp_tclon1 = []
    interp_trans_spd1 = []
    for t in range(len(tc)-1): #마지막 스텝이 하나 짤림
        d1= datetime.datetime(int(tc.yr.iloc[t]), int(tc.mo.iloc[t]), int(tc.dy.iloc[t]), int(tc.hr.iloc[t][0:2]))
        d2 = datetime.datetime(int(tc.yr.iloc[t+1]), int(tc.mo.iloc[t+1]), int(tc.dy.iloc[t+1]), int(tc.hr.iloc[t+1][0:2]))
        d3 = d2 - d1
        delta_hrs = int(d3.total_seconds() / 3600) #초를 시간으로

        #마지막 스텝이면 delta_hrs+1 해서 마지막 짤림방지
        if t == len(tc)-2:
            interp_tclat1.append(list(interp_method3(tc.lat[t:t+2], delta_hrs+1)) )
            interp_tclon1.append(list(interp_method3(tc.lon[t:t+2], delta_hrs+1)) )

            interp_trans_spd1.append(list(interp_method3(tc.trans_spd[t:t+2], delta_hrs+1)) )
        else:
            interp_tclat1.append(list(interp_method3(tc.lat[t:t+2], delta_hrs)) )
            interp_tclon1.append(list(interp_method3(tc.lon[t:t+2], delta_hrs)) )

            interp_trans_spd1.append(list(interp_method3(tc.trans_spd[t:t+2], delta_hrs)) )

    interp_tclat = np.sum(interp_tclat1)
    interp_tclon = np.sum(interp_tclon1)
    interp_trans_spd = np.sum(interp_trans_spd1)




    for p in range(1,len(varnames)):
        globals()['mean_'+varnames[p]+'_'+str(i)] = np.zeros([len(interp_tclat)])
        globals()['max_'+varnames[p]+'_'+str(i)] = np.zeros([len(interp_tclat)])   
    globals()['min_'+varnames[0]+'_'+str(i)] = np.zeros([len(interp_tclat)])   


    k_idx= -1
    tcdf = pd.DataFrame([])
    for k in range(len(interp_tclat)):
        k_idx += 1
        lnlt_idx = []
        ln_idx = []
        lt_idx = []

        #map_lon, map_lat 다돌지말고, 주변한 N도만 돌면 되잖아
        map_ln_idx = np.where(map_lon == find_nearest(map_lon, interp_tclon[k]))[0][0]
        map_lt_idx = np.where(map_lat == find_nearest(map_lat, interp_tclat[k]))[0][0]

        '''
        <기상청>
        단계	풍속 15m/s 이상의 반경
            소형	300km 미만
            중형	300km 이상 ~ 500km 미만
        대형	500km 이상 ~ 800km 미만
        초대형	800km 이상
        '''
        steps = map_lon[2] - map_lon[1] #격자 간격
        around_degree = 3          #주변 몇도만 탐색?
        step_idx = int(around_degree/steps) # 몇도 탐색을 위한 인덱스 개수


        for lns, lts in itertools.product(range(map_ln_idx-step_idx, map_ln_idx+step_idx+1), range(map_lt_idx-step_idx, map_lt_idx+step_idx+1)):
            distance = GeoUtil.get_harversine_distance(interp_tclon[k],interp_tclat[k],map_lon[lns],map_lat[lts])
            if distance <= around_degree:
                ln_idx.append(lns)     
                lt_idx.append(lts)     

        #ERA5 데이터들을 가져와서 새로운 tc dataframe을 만들자.
        for p in range(1,len(varnames)):
            eval('mean_'+varnames[p]+'_'+str(i))[k] = np.mean(eval(varnames[p])[k,lt_idx, ln_idx])
            eval('max_'+varnames[p]+'_'+str(i))[k] = np.max(eval(varnames[p])[k,lt_idx, ln_idx])
        eval('min_'+varnames[0]+'_'+str(i))[k] = np.min(eval(varnames[0])[k,lt_idx, ln_idx])



        #1시간 interp한 새로운 tc dataframe
        if k == 0:
            df_hr = start_hr
            df_dy = start_dy
            df_mo = start_mo
        if df_hr == 24:
            df_dy += 1
            df_hr = 0
            start_hr = 0
            if df_mo ==7 or df_mo ==8 or df_mo ==10:
                if df_dy == 32:
                    df_mo +=1
                    df_dy = 1
            if df_mo ==6 or df_mo==9:
                if df_dy == 31:
                    df_mo +=1   
                    df_dy = 1

        df_data = [tc.idx[0],  #고정
                        tc.num[0],  #고정
                        start_yr, #고정
                        df_mo,  
                        df_dy, 
                        df_hr, 
                        round(interp_tclon[k] ,2), 
                        round(interp_tclat[k] ,2), 
                        round(eval('min_mslp_'+str(i))[k] ,2), 
                        round(eval('max_wspd10_'+str(i))[k] ,2), 
                        round(eval('max_wspd100_'+str(i))[k] ,2), 
                        round(eval('max_wspd850_'+str(i))[k] ,2), 
                        round(eval('mean_tp_'+str(i))[k] ,2),
                        round(interp_trans_spd[k] ,2)]
        df_hr += 1
        tcdf = pd.concat([tcdf, pd.DataFrame([df_data])], ignore_index=True)
        
    tcdf.columns=['idx','num','yr','mo','dy','hr','lon','lat','mslp','max_wspd10','max_wspd100','max_wspd850','mean_tp','trans_spd']    
    globals()['interp_tc_'+str(i)] = tcdf











#plot
for idx in range(total_tcs):

    tc = KMI_data(idx)
    fig, ax1 = plt.subplots(1,1,figsize=(8,5))

    interp_tcobs1 = []
    for t in range(len(tc)-1): #마지막 스텝이 하나 짤림
        d1= datetime.datetime(int(tc.yr.iloc[t]), int(tc.mo.iloc[t]), int(tc.dy.iloc[t]), int(tc.hr.iloc[t][0:2]))
        d2 = datetime.datetime(int(tc.yr.iloc[t+1]), int(tc.mo.iloc[t+1]), int(tc.dy.iloc[t+1]), int(tc.hr.iloc[t+1][0:2]))
        d3 = d2 - d1
        delta_hrs = int(d3.total_seconds() / 3600) #초를 시간으로
        #마지막 스텝이면 delta_hrs+1 해서 마지막 짤림방지
        if t == len(tc)-2:
            interp_tcobs1.append(list(interp_method3(tc.spd[t:t+2], delta_hrs+1)) )
        else:
            interp_tcobs1.append(list(interp_method3(tc.spd[t:t+2], delta_hrs)) )
    interp_tcobs = np.sum(interp_tcobs1)


    ax1.plot(interp_tcobs, '-b', label ='Obs(KMI) wind speed')
    ax1.plot(np.array(eval('interp_tc_'+str(idx)).max_wspd10) , '--r', label ='ERA5 max 10m wspd')
    ax1.plot(np.array(eval('interp_tc_'+str(idx)).max_wspd100) , '--g', label ='ERA5 max 100m wspd') 
    ax1.plot(np.array(eval('interp_tc_'+str(idx)).max_wspd850) , '--k', label ='ERA5 max 850m wspd') 
    ax1.set_ylim([0, 60])

    ax1.legend(loc='best', fontsize=10)


    if idx == 4 or idx == 6:
        ax1.set_title('Wind speed in '+str(around_degree)+'° from TC center ('+str(idx)+'), (-36h)', fontweight='bold', fontsize=12)
        plt.savefig('C:/Users/rjsdn/Desktop/KMI/Wind_speed_'+str(idx)+'.png', bbox_inches='tight')

    else:
        ax1.set_title('Wind speed in '+str(around_degree)+'° from TC center ('+str(idx)+')', fontweight='bold', fontsize=12)
        plt.savefig('C:/Users/rjsdn/Desktop/KMI/Wind_speed_'+str(idx)+'.png', bbox_inches='tight')







#plot
#strong_jet_idx
#mod_jet_idx
#weak_jet_idx
for idx in list(weak_jet_idx):
    tc = KMI_data(idx)
    fig, ax1 = plt.subplots(1,1,figsize=(8,5))

    ax1.plot(np.array(eval('interp_tc_'+str(idx)).mean_tp) , '-b')
    ax1.set_ylabel("mm",color="b",fontsize=14)
    ax1.set_ylim([0, 5])

    ax2=ax1.twinx()
    ax2.plot(np.array(eval('interp_tc_'+str(idx)).trans_spd) , '-m')
    ax2.set_ylabel("km/h",color="m",fontsize=14)
    ax2.set_ylim([0, 100])

    #custom legend
    custom_lines = [Line2D([0], [0], color='b', lw=4),
                    Line2D([0], [0], color='m', lw=4)
                    ]
    plt.legend(custom_lines, ['ERA5 total precipitaion', 'KMI translation speed'], loc='upper left')

    ax1.set_title('TP&TS '+str(idx), fontweight='bold', fontsize=12)
    plt.savefig('C:/Users/rjsdn/Desktop/KMI/TP&TS_'+str(idx)+'.png', bbox_inches='tight')


















#mslp_4의 경우 36시간전 데이터를 하면 베스트트랙과 잘 맞음.
for i in range(0,180,24):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    map = Basemap(projection='mill',llcrnrlat=0,urcrnrlat=50,llcrnrlon=120,urcrnrlon=180)
    map.drawcoastlines()
    map.drawmeridians(np.arange(0, 360, 30), labels=[0,0,0,1])
    map.drawparallels(np.arange(-90, 90, 10), labels=[1,0,0,0])

    x,y = map(map_lon,map_lat) 
    plt.contourf(x, y, mslp_4[i,:,:], colormap='RdBu_r')

    x,y = map(interp_tc_4.lon[i], interp_tc_4.lat[i]) #베스트트랙 값
    plt.plot(x, y, '-ro', markersize=4)

    plt.colorbar()







#mslp_6의 경우 36시간전 데이터를 하면 베스트트랙과 잘 맞음.
for i in range(0,97,12):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    map = Basemap(projection='mill',llcrnrlat=15,urcrnrlat=50,llcrnrlon=120,urcrnrlon=150)
    map.drawcoastlines()
    map.drawmeridians(np.arange(0, 360, 30), labels=[0,0,0,1])
    map.drawparallels(np.arange(-90, 90, 10), labels=[1,0,0,0])

    x,y = map(map_lon,map_lat) 
    plt.contourf(x, y, mslp_6[i,:,:], colormap='RdBu_r')

    x,y = map(interp_tc_6.lon[i], interp_tc_6.lat[i]) #베스트트랙 값
    plt.plot(x, y, '-ro', markersize=8)

    plt.colorbar()






TC_loc_verification(7)
#mslp_7의 경우 36시간전 데이터를 하면 베스트트랙과 잘 맞음.
for i in range(0,175,24):
    fig, ax = plt.subplots(1,1, figsize=(10,10))
    map = Basemap(projection='mill',llcrnrlat=15,urcrnrlat=50,llcrnrlon=120,urcrnrlon=150)
    map.drawcoastlines()
    map.drawmeridians(np.arange(0, 360, 30), labels=[0,0,0,1])
    map.drawparallels(np.arange(-90, 90, 10), labels=[1,0,0,0])

    x,y = map(map_lon,map_lat) 
    plt.contourf(x, y, mslp_7[i,:,:], colormap='RdBu_r')

    x,y = map(interp_tc_7.lon[i], interp_tc_7.lat[i]) #베스트트랙 값
    plt.plot(x, y, '-ro', markersize=8)

    plt.colorbar()
