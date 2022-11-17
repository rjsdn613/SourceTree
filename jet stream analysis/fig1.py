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

##################################################################

ao_data = pd.read_csv('E:/CSL/AO_index.txt', sep = "\s+")
ao_data.insert(3,'TC_number',0)
ao_data.insert(4,'TCs',0)
ao_data=ao_data.reset_index()
ao_data=ao_data.drop('index',axis=1)

a=-1
negative_ao = np.zeros([480,3])
positive_ao = np.zeros([480,3])

for i in range(480):
    a+=1
    if ao_data.ao[i] < 0:
        negative_ao[a,0]=ao_data.year[i]
        negative_ao[a,1]=ao_data.mo[i]
        negative_ao[a,2]=ao_data.ao[i]
    if ao_data.ao[i] > 0:
        positive_ao[a,0]=ao_data.year[i]
        positive_ao[a,1]=ao_data.mo[i]
        positive_ao[a,2]=ao_data.ao[i]



nn=pd.DataFrame(negative_ao)
null = nn[ nn[0] == 0 ].index
neg_ao_Data=nn.drop(null)
neg_ao_Data.columns=["yr", "mo", "AO_idx"]
neg_ao_Data.astype({'yr':int,'mo':int,'AO_idx':np.float64})
neg_ao_Data.insert(3,'TC_number',0)
neg_ao_Data.insert(4,'TCs',0)
neg_ao_Data=neg_ao_Data.reset_index()
neg_ao_Data=neg_ao_Data.drop('index',axis=1)


nn=pd.DataFrame(positive_ao)
null = nn[ nn[0] == 0 ].index
pos_ao_Data=nn.drop(null)
pos_ao_Data.columns=["yr", "mo", "AO_idx"]
pos_ao_Data.astype({'yr':int,'mo':int,'AO_idx':np.float64})
pos_ao_Data.insert(3,'TC_number',0)
pos_ao_Data.insert(4,'TCs',0)
pos_ao_Data=pos_ao_Data.reset_index()
pos_ao_Data=pos_ao_Data.drop('index',axis=1)


############ 태풍시즌 데이터만 새로 추출 ##########################

TCseason_pos_ao = pos_ao_Data.loc[pos_ao_Data[ (pos_ao_Data['mo'] == 8) | \
     (pos_ao_Data['mo'] == 9) | (pos_ao_Data['mo'] == 10) ].index]

TCseason_neg_ao = neg_ao_Data.loc[neg_ao_Data[ (neg_ao_Data['mo'] == 8) | \
     (neg_ao_Data['mo'] == 9) | (neg_ao_Data['mo'] == 10) ].index]

TCseason_neg_ao = TCseason_neg_ao.reset_index()
TCseason_neg_ao=TCseason_neg_ao.drop('index',axis=1)

TCseason_pos_ao = TCseason_pos_ao.reset_index()
TCseason_pos_ao=TCseason_pos_ao.drop('index',axis=1)

TCseason_ao = pd.concat([TCseason_neg_ao,TCseason_pos_ao]) # pos+neg 
TCseason_ao = TCseason_ao.sort_values(['yr','mo'], axis=0, ascending=True, inplace=False) # 시간순 정렬
TCseason_ao = TCseason_ao.reset_index() 
TCseason_ao=TCseason_ao.drop('index',axis=1)

##################################################################
# 태풍의 발생 월 과 AO_idx 가 음(-)일때 의 월이 같으면 해당 태풍을 선정.
with open("E:/CSL/new_bst_all_80.txt", "r") as f:

    yr = []
    mo = []
    TCnum = []

    line = f.readline()
    TC_info_line = line.split()
    TC_number = int(TC_info_line[1])
    TC_count_num = int(TC_info_line[2])
    TCnum.append(TC_number)
    data = f.readline()
    date = int(data.split()[0])
    ayr = int(date * 0.000001)
    amo = int(date * 0.0001) - (ayr * 100)
    yr.append(ayr)
    mo.append(amo)

    a = 0
    while True:
        a += 1
        if a > 1:
            line = f.readline()
            if not line:
                break
            TC_info_line = line.split()
            TC_number = int(TC_info_line[1])
            TCnum.append(TC_number)
            TC_count_num = int(TC_info_line[2])

            data = f.readline()
            date = int(data.split()[0])
            ayr = int(date * 0.000001)
            amo = int(date * 0.0001) - (ayr * 100)
            #ady = int(date * 0.01) - int(date * 0.0001) * 100
            #ahr = date % 100

            yr.append(ayr)
            mo.append(amo)
        for i in range(TC_count_num-1):
            f.readline()         

###########################################################################
for j in range(59):
    test=[]
    for i in range(1030):
        if (TCseason_neg_ao.loc[j,'yr'] % 100) == yr[i] and TCseason_neg_ao.loc[j,'mo'] == mo[i]:
            test.append(TCnum[i])
            TCseason_neg_ao.loc[j,'TC_number'] = str(test)
    del(test)

for j in range(59):
    if TCseason_neg_ao.loc[j,'TC_number'] != 0:
        TCseason_neg_ao.loc[j,'TCs'] = len(TCseason_neg_ao.loc[j,'TC_number'].split(','))
##########################

for j in range(61):
    test=[]
    for i in range(1030):
        if (TCseason_pos_ao.loc[j,'yr'] % 100) == yr[i] and TCseason_pos_ao.loc[j,'mo'] == mo[i]:
            test.append(TCnum[i])
            TCseason_pos_ao.loc[j,'TC_number'] = str(test)
    del(test)

for j in range(61):
    if TCseason_pos_ao.loc[j,'TC_number'] != 0:
        TCseason_pos_ao.loc[j,'TCs'] = len(TCseason_pos_ao.loc[j,'TC_number'].split(','))  

##########################
for j in range(120):
    test=[]
    for i in range(1030):
        if (TCseason_ao.loc[j,'yr'] % 100) == yr[i] and TCseason_ao.loc[j,'mo'] == mo[i]:
            test.append(TCnum[i])
            TCseason_ao.loc[j,'TC_number'] = str(test)
    del(test)

for j in range(120):
    if TCseason_ao.loc[j,'TC_number'] != 0:
        TCseason_ao.loc[j,'TCs'] = len(TCseason_ao.loc[j,'TC_number'].split(','))  

##########################################################################

'''
데이터 전처리 완료 (양, 음 북극진동 월 별로 태풍 뭐뭐있는지 갯수 몇개인지.,  pos_ao_Data, neg_ao_Data
'''

##############################################
'''
1. AO index time series + 같은타임에 태풍갯수
2. AO 음 양일때 평균 제트 위치 그림, 제트핵위치(가장 풍속강한 그리드포인트의 위도) 차이가 통계적을 유의미한지 확인
3. AO 음 양일때 평균 제트위치 그림 + 태풍최대강도의 평균위치(AO 음, 양 나눠서) and 태풍마지막 기록 평균위치(AO 음 양 나눠서)
    제트위치 상하에 따라 태풍의 LMI 위치나 최대로올라올수있는 지점이 달라짐을 주장하려면, SST VWS 도 같이 그려서 비교해줘야 됨

    noaa.ersst.v5 , 1525 time 부터 81년 1월 , 2x2 degree , montly mean 
    lat = 89 ;
    lon = 180 ;
    sst(time, lat, lon) ;
    88.0N - 88.0S, 0.0E - 358.0E.
'''
##########################################################################################################
# 1. AO index time series + 태풍갯수 plot
##########################################################################################################
# AO=pd.DataFrame(ao_data)
# # AO.TCs[AO.TCs[ AO.TCs == 0 ].index] = np.NaN

# # fig, ax1 =  plt.subplots()
# # ax1.plot(AO.ao, linestyle='-',color='grey',linewidth=2)
# # plt.xticks(np.arange(0, 481, 12*5), labels=['1981', '1986', '1991', '1996', '2001', '2006', '2011','2016','2021'])
# # plt.yticks([-4,-3,-2,-1,0,1,2,3,4])
# # plt.axhline(y=0, color='k', linewidth=2,linestyle='--')
# # plt.grid(axis='x')

# # ax2 = ax1.twinx()
# # ax2.plot(AO.TCs,linestyle='-', color='lightblue',linewidth=2)


# # plt.show()


# corr = np.corrcoef(AO.ao,AO.TCs )[0, 1] # -0.01

'''
AO index 값의 크기와 같은 타임의 태풍 발생 갯수는 상관이 없다.
'''
##########################################################################################################
#2. AO 음 양일때 평균 제트 위치 그림, 제트핵위치(가장 풍속강한 그리드포인트의 위도) 차이가 통계적을 유의미한지 확인
##########################################################################################################

##############################################################1. 양의 북극진동인 월 평균 
pyr = TCseason_pos_ao.yr
pmo = TCseason_pos_ao.mo
p_avg_u = np.zeros([73,144,len(pyr)])

for i in range(len(pyr)):
    uwnd_file = "E:/CSL/NCEP/reanalysis data/uwnd."+str(int(pyr[i]))+".nc"
    uwnd_filedata = nc.Dataset(uwnd_file)
    uwnd_filedata.variables.keys()

    lon = uwnd_filedata.variables["lon"][:]
    lat = uwnd_filedata.variables["lat"][:]
    # uwnd = uwnd_filedata.variables["uwnd"][:,:,:,:]  # time level lat lon
    # time = uwnd_filedata.variables["time"][:]
    # level = uwnd_filedata.variables["level"][:] # 9 : 200hPa 

    # 1,3,5,7,8,10,12 월 은 31일 ; 2월은 28(29)일 ; 4,6,9,11 월은 30일

    motime=[[1,31],[32,59],[60,90],[91,120],[121,151],[152,181],[182,212],[213,243]\
        ,[244,273],[274,304],[305,334],[335,365]]

    if pmo[i] == 1:
        uwnd = np.mean(np.array(uwnd_filedata.variables["uwnd"][motime[0][0]:motime[0][1],9,:,:]),0)

    for idx in range(2,13):
        if pmo[i] == idx and pyr[i] % 4 != 0:
            uwnd = np.mean(np.array(uwnd_filedata.variables["uwnd"][motime[idx-1][0]:motime[idx-1][1],9,:,:]),0)
        elif pmo[i] == idx and pyr[i] % 4 == 0 and pyr[i] != 2000:
            uwnd = np.mean(np.array(uwnd_filedata.variables["uwnd"][motime[idx-1][0]:motime[idx-1][1]+1,9,:,:]),0)

    p_avg_u[:,:,i] = uwnd

positive_avg_u = np.mean(p_avg_u,2)



### 변수 초기화
del(uwnd_filedata)
del(uwnd_file)
###


##############################################################1. 음의 북극진동인 월 평균 
nyr = TCseason_neg_ao.yr
nmo = TCseason_neg_ao.mo
n_avg_u = np.zeros([73,144,len(nyr)])

for i in range(len(nyr)):
    uwnd_file = "E:/CSL/NCEP/reanalysis data/uwnd."+str(int(nyr[i]))+".nc"
    uwnd_filedata = nc.Dataset(uwnd_file)
    uwnd_filedata.variables.keys()

    lon = uwnd_filedata.variables["lon"][:]
    lat = uwnd_filedata.variables["lat"][:]
    # uwnd = uwnd_filedata.variables["uwnd"][:,:,:,:]  # time level lat lon
    # time = uwnd_filedata.variables["time"][:]
    # level = uwnd_filedata.variables["level"][:] # 9 : 200hPa 

    # 1,3,5,7,8,10,12 월 은 31일 ; 2월은 28(29)일 ; 4,6,9,11 월은 30일

    motime=[[1,31],[32,59],[60,90],[91,120],[121,151],[152,181],[182,212],[213,243]\
        ,[244,273],[274,304],[305,334],[335,365]]

    if nmo[i] == 1:
        uwnd = np.mean(np.array(uwnd_filedata.variables["uwnd"][motime[0][0]:motime[0][1],9,:,:]),0)

    for idx in range(2,13):
        if nmo[i] == idx and nyr[i] % 4 != 0:
            uwnd = np.mean(np.array(uwnd_filedata.variables["uwnd"][motime[idx-1][0]:motime[idx-1][1],9,:,:]),0)
        elif nmo[i] == idx and nyr[i] % 4 == 0 and nyr[i] != 2000:
            uwnd = np.mean(np.array(uwnd_filedata.variables["uwnd"][motime[idx-1][0]:motime[idx-1][1]+1,9,:,:]),0)

    n_avg_u[:,:,i] = uwnd

negative_avg_u = np.mean(n_avg_u,2)



### 변수 초기화
del(uwnd_filedata)
del(uwnd_file)
###


###########################################################################################
#plot
###########################################################################################

fig = plt.figure()

ax = fig.add_subplot(1,2,1)
map = Basemap(
    projection="mill",
    lon_0=180,
    # llcrnrlon=100,
    # llcrnrlat=10,
    # urcrnrlon=360,
    # urcrnrlat=60,
    resolution="l",
)

lons, lats = meshgrid(lon, lat)
x, y = map(lons, lats)

# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color='w',alpha=0.01)
map.drawcoastlines()
map.contourf(x, y, positive_avg_u,range(-20,80,10),cmap="hot_r")

# draw lat lon label on map
# map.drawparallels(np.arange(int(10), int(60), 10), labels=[1, 0, 0, 0],fontsize=7)
# map.drawmeridians(np.arange(int(100), int(360), 50), labels=[0, 0, 0, 1],fontsize=7)

plt.title("positive_avg_u in TC season",fontsize=7,fontweight='bold')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=7)

"""
orientation (horizontal or vertical)
fraction (default: 0.15. colorbar가 차지하는 영역의 비율)
aspect (default: 20. colorbar의 긴 변 : 짧은 변 비율)
"""


ax = fig.add_subplot(1,2,2)
map = Basemap(
    projection="mill",
    lon_0=180,
    # llcrnrlon=100,
    # llcrnrlat=10,
    # urcrnrlon=360,
    # urcrnrlat=60,
    resolution="l",
)

# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color='w',alpha=0.01)
map.drawcoastlines()
map.contourf(x, y, negative_avg_u,range(-20,80,10), cmap="hot_r")

# draw lat lon label on map
# map.drawparallels(np.arange(int(10), int(60), 10), labels=[1, 0, 0, 0],fontsize=7)
# map.drawmeridians(np.arange(int(100), int(360), 50), labels=[0, 0, 0, 1],fontsize=7)

plt.title("negative_avg_u in TC season",fontsize=7,fontweight='bold')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=7)


# plt.suptitle("",fontsize=20,fontweight='bold')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.1, wspace=0.3)
plt.savefig("E:/CSL/visual studio/SourceTree/jet stream analysis/fig1_mill.png", dpi=600)
plt.close()
# plt.show()



#########################################
# merc NWP plot
############################
fig = plt.figure()

ax = fig.add_subplot(1,2,1)
map = Basemap(
    projection="merc",
    llcrnrlon=80,
    llcrnrlat=0,
    urcrnrlon=360,
    urcrnrlat=70,
    resolution="l",
)

lons, lats = meshgrid(lon, lat)
x, y = map(lons, lats)

# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color='w',alpha=0.01)
map.drawcoastlines()
map.contourf(x, y, positive_avg_u,range(-20,80,10),cmap="hot_r")

# draw lat lon label on map
# map.drawparallels(np.arange(int(10), int(60), 10), labels=[1, 0, 0, 0],fontsize=7)
# map.drawmeridians(np.arange(int(100), int(360), 50), labels=[0, 0, 0, 1],fontsize=7)

plt.title("positive_avg_u in TC season",fontsize=7,fontweight='bold')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=5)

"""
orientation (horizontal or vertical)
fraction (default: 0.15. colorbar가 차지하는 영역의 비율)
aspect (default: 20. colorbar의 긴 변 : 짧은 변 비율)
"""


ax = fig.add_subplot(1,2,2)



map1 = Basemap(
    projection="merc",
    llcrnrlon=80,
    llcrnrlat=0,
    urcrnrlon=360,
    urcrnrlat=70,
    resolution="l",
)


lons, lats = meshgrid(lon, lat)
x, y = map1(lons, lats)




# map.drawmapboundary(fill_color='aqua').
map1.fillcontinents(color='w',alpha=0.01)
map1.drawcoastlines()
map1.contourf(x, y, negative_avg_u,range(-20,80,10), cmap="hot_r")

# draw lat lon label on map
# map.drawparallels(np.arange(int(10), int(60), 10), labels=[1, 0, 0, 0],fontsize=7)
# map.drawmeridians(np.arange(int(100), int(360), 50), labels=[0, 0, 0, 1],fontsize=7)

plt.title("negative_avg_u in TC season",fontsize=7,fontweight='bold')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=5)


# plt.suptitle("",fontsize=20,fontweight='bold')
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.1, wspace=0.3)
# plt.show()





plt.savefig("E:/CSL/visual studio/SourceTree/jet stream analysis/fig1_Mid.png", dpi=600)
plt.close()