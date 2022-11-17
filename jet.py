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
for j in range(237):
    test=[]
    for i in range(1030):
        if (neg_ao_Data.loc[j,'yr'] % 100) == yr[i] and neg_ao_Data.loc[j,'mo'] == mo[i]:
            test.append(TCnum[i])
            neg_ao_Data.loc[j,'TC_number'] = str(test)
    del(test)

for j in range(237):
    if neg_ao_Data.loc[j,'TC_number'] != 0:
        neg_ao_Data.loc[j,'TCs'] = len(neg_ao_Data.loc[j,'TC_number'].split(','))


for j in range(243):
    test=[]
    for i in range(1030):
        if (pos_ao_Data.loc[j,'yr'] % 100) == yr[i] and pos_ao_Data.loc[j,'mo'] == mo[i]:
            test.append(TCnum[i])
            pos_ao_Data.loc[j,'TC_number'] = str(test)
    del(test)

for j in range(243):
    if pos_ao_Data.loc[j,'TC_number'] != 0:
        pos_ao_Data.loc[j,'TCs'] = len(pos_ao_Data.loc[j,'TC_number'].split(','))  


for j in range(480):
    test=[]
    for i in range(1030):
        if (ao_data.loc[j,'year'] % 100) == yr[i] and ao_data.loc[j,'mo'] == mo[i]:
            test.append(TCnum[i])
            ao_data.loc[j,'TC_number'] = str(test)
    del(test)

for j in range(480):
    if ao_data.loc[j,'TC_number'] != 0:
        ao_data.loc[j,'TCs'] = len(ao_data.loc[j,'TC_number'].split(','))  

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
pyr = pos_ao_Data.yr
pmo = pos_ao_Data.mo
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

################################################################2. 양의 북극진동인 월 중에 태풍있는 월만 뽑아서 평균 
null = pos_ao_Data[ pos_ao_Data.TCs == 0 ].index
null_pos_ao_Data=pos_ao_Data.drop(null) #169개
null_pos_ao_Data=null_pos_ao_Data.reset_index()
null_pos_ao_Data=null_pos_ao_Data.drop('index',axis=1)

pyr = null_pos_ao_Data.yr
pmo = null_pos_ao_Data.mo
p_avg_u = np.zeros([73,144,len(pyr)])

for i in range(len(pyr)):
    uwnd_file = "E:/CSL/NCEP/reanalysis data/uwnd."+str(int(pyr[i]))+".nc"
    uwnd_filedata = nc.Dataset(uwnd_file)
    uwnd_filedata.variables.keys()

    lon = uwnd_filedata.variables["lon"][:]
    lat = uwnd_filedata.variables["lat"][:]

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

null_positive_avg_u = np.mean(p_avg_u,2)


##############################################################1. 음의 북극진동인 월 평균 
nyr = neg_ao_Data.yr
nmo = neg_ao_Data.mo
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

################################################################2. 음의 북극진동인 월 중에 태풍있는 월만 뽑아서 평균 
null = neg_ao_Data[ neg_ao_Data.TCs == 0 ].index
null_neg_ao_Data=neg_ao_Data.drop(null) #169개
null_neg_ao_Data=null_neg_ao_Data.reset_index()
null_neg_ao_Data=null_neg_ao_Data.drop('index',axis=1)

nyr = null_neg_ao_Data.yr
nmo = null_neg_ao_Data.mo
n_avg_u = np.zeros([73,144,len(nyr)])

for i in range(len(nyr)):
    uwnd_file = "E:/CSL/NCEP/reanalysis data/uwnd."+str(int(nyr[i]))+".nc"
    uwnd_filedata = nc.Dataset(uwnd_file)
    uwnd_filedata.variables.keys()

    lon = uwnd_filedata.variables["lon"][:]
    lat = uwnd_filedata.variables["lat"][:]

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

null_negative_avg_u = np.mean(n_avg_u,2)




###########################################################################################
#plot
###########################################################################################

fig = plt.figure()

ax = fig.add_subplot(2,2,1)
map = Basemap(
    projection="merc",
    llcrnrlon=100,
    llcrnrlat=10,
    urcrnrlon=360,
    urcrnrlat=60,
    resolution="h",
)
lons, lats = np.meshgrid(lon, lat)
x, y = map(lons, lats)
# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color='w',alpha=0.01)
map.drawcoastlines()
map.contourf(x, y, positive_avg_u,cmap="hot_r")
# draw lat lon label on map
map.drawparallels(np.arange(int(10), int(60), 10), labels=[1, 0, 0, 0],fontsize=7)
map.drawmeridians(np.arange(int(100), int(360), 50), labels=[0, 0, 0, 1],fontsize=7)

plt.title("all_positive_avg_u",fontsize=7,fontweight='bold')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=7)

"""
orientation (horizontal or vertical)
fraction (default: 0.15. colorbar가 차지하는 영역의 비율)
aspect (default: 20. colorbar의 긴 변 : 짧은 변 비율)
"""


ax = fig.add_subplot(2,2,2)
map = Basemap(
    projection="merc",
    llcrnrlon=100,
    llcrnrlat=10,
    urcrnrlon=360,
    urcrnrlat=60,
    resolution="h",
)
# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color='w',alpha=0.01)
map.drawcoastlines()
map.contourf(x, y, null_positive_avg_u, cmap="hot_r")

# draw lat lon label on map
map.drawparallels(np.arange(int(10), int(60), 10), labels=[1, 0, 0, 0],fontsize=7)
map.drawmeridians(np.arange(int(100), int(360), 50), labels=[0, 0, 0, 1],fontsize=7)

plt.title("positive_avg_u",fontsize=7,fontweight='bold')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=7)


ax = fig.add_subplot(2,2,3)
map = Basemap(
    projection="merc",
    llcrnrlon=100,
    llcrnrlat=10,
    urcrnrlon=360,
    urcrnrlat=60,
    resolution="h",
)
# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color='w',alpha=0.01)
map.drawcoastlines()
map.contourf(x, y, negative_avg_u, cmap="hot_r")

# draw lat lon label on map
map.drawparallels(np.arange(int(10), int(60), 10), labels=[1, 0, 0, 0],fontsize=7)
map.drawmeridians(np.arange(int(100), int(360), 50), labels=[0, 0, 0, 1],fontsize=7)

plt.title("all_negative_avg_u",fontsize=7,fontweight='bold')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=7)


ax = fig.add_subplot(2,2,4)
map = Basemap(
    projection="merc",
    llcrnrlon=100,
    llcrnrlat=10,
    urcrnrlon=360,
    urcrnrlat=60,
    resolution="h",
)
# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color='w',alpha=0.01)
map.drawcoastlines()
map.contourf(x, y, null_negative_avg_u, cmap="hot_r")

# draw lat lon label on map
map.drawparallels(np.arange(int(10), int(60), 10), labels=[1, 0, 0, 0],fontsize=7)
map.drawmeridians(np.arange(int(100), int(360), 50), labels=[0, 0, 0, 1],fontsize=7)

plt.title("negative_avg_u",fontsize=7,fontweight='bold')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=7)



# plt.suptitle("",fontsize=20,fontweight='bold')
plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.1, wspace=0.3)
plt.savefig("E:/CSL/visual studio/SourceTree/jet stream analysis/fig1.png", dpi=600)
plt.close()
# plt.show()













##########################################################################################################
# AOI min, max 의 경우 
pos_ao_Data.loc[pos_ao_Data[pos_ao_Data.AO_idx == np.max(pos_ao_Data.AO_idx)].index]
neg_ao_Data.loc[neg_ao_Data[neg_ao_Data.AO_idx == np.min(neg_ao_Data.AO_idx)].index]
### negative minimum(2010년 2월 , AOI = -4.2657) , positive maximum(93년 1월 , AOI = 3.4953)

uwnd_file = "E:/CSL/NCEP/reanalysis data/uwnd.2010.nc"
uwnd_filedata = nc.Dataset(uwnd_file)
uwnd_filedata.variables.keys()
uwnd = np.mean(np.array(uwnd_filedata.variables["uwnd"][32:59,9,:,:]),0)
min_aoi_u  = uwnd


uwnd_file = "E:/CSL/NCEP/reanalysis data/uwnd.1993.nc"
uwnd_filedata = nc.Dataset(uwnd_file)
uwnd_filedata.variables.keys()
uwnd = np.mean(np.array(uwnd_filedata.variables["uwnd"][1:31,9,:,:]),0)
max_aoi_u  = uwnd


fig = plt.figure()

ax = fig.add_subplot(1,2,1)
map = Basemap(
    projection="merc",
    llcrnrlon=100,
    llcrnrlat=10,
    urcrnrlon=360,
    urcrnrlat=60,
    resolution="h",
)
lons, lats = np.meshgrid(lon, lat)
x, y = map(lons, lats)
# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color='w',alpha=0.01)
map.drawcoastlines()
map.contourf(x, y, min_aoi_u,range(-20,100,10),cmap="hot_r")
# draw lat lon label on map
map.drawparallels(np.arange(int(10), int(60), 10), labels=[1, 0, 0, 0],fontsize=7)
map.drawmeridians(np.arange(int(100), int(360), 50), labels=[0, 0, 0, 1],fontsize=7)

plt.title("min_aoi_u",fontsize=7,fontweight='bold')



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
    projection="merc",
    llcrnrlon=100,
    llcrnrlat=10,
    urcrnrlon=360,
    urcrnrlat=60,
    resolution="h",
)
# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color='w',alpha=0.01)
map.drawcoastlines()
map.contourf(x, y, max_aoi_u,range(-20,100,10), cmap="hot_r")

# draw lat lon label on map
map.drawparallels(np.arange(int(10), int(60), 10), labels=[1, 0, 0, 0],fontsize=7)
map.drawmeridians(np.arange(int(100), int(360), 50), labels=[0, 0, 0, 1],fontsize=7)


plt.title("max_aoi_u",fontsize=7,fontweight='bold')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=7)



fig.tight_layout()
plt.savefig("E:/CSL/visual studio/SourceTree/jet stream analysis/fig2.png", dpi=600)
plt.close()
# plt.show()




#####################################################
# JJA , DJF 평균 AOI 값과 제트기류
jja = pd.DataFrame()
djf = pd.DataFrame()
for i in range(len(ao_data)):
    if ao_data.mo[i] == 6 or ao_data.mo[i] == 7 or ao_data.mo[i] == 8:
        jja=jja.append(ao_data.loc[i], ignore_index=True)
    elif ao_data.mo[i] == 11 or ao_data.mo[i] == 12 or ao_data.mo[i] == 1:
        djf=djf.append(ao_data.loc[i], ignore_index=True)    

# np.mean(jja.ao) = -0.047
# np.mean(djf.ao) = 0.030

# 태풍없는 월 제외
null = jja[ jja.TCs == 0 ].index
jja=jja.drop(null) #
jja=jja.reset_index()
jja=jja.drop('index',axis=1)

null = djf[ djf.TCs == 0 ].index
djf=djf.drop(null) 
djf=djf.reset_index()
djf=djf.drop('index',axis=1)



jja_yr = jja.year
jja_mo = jja.mo
jja_avg_u = np.zeros([73,144,len(jja_yr)])



for i in range(len(jja_yr)):
    uwnd_file = "E:/CSL/NCEP/reanalysis data/uwnd."+str(int(jja_yr[i]))+".nc"
    uwnd_filedata = nc.Dataset(uwnd_file)
    uwnd_filedata.variables.keys()

    lon = uwnd_filedata.variables["lon"][:]
    lat = uwnd_filedata.variables["lat"][:]

    motime=[[1,31],[32,59],[60,90],[91,120],[121,151],[152,181],[182,212],[213,243]\
        ,[244,273],[274,304],[305,334],[335,365]]

    if jja_mo[i] == 1:
        uwnd = np.mean(np.array(uwnd_filedata.variables["uwnd"][motime[0][0]:motime[0][1],9,:,:]),0)

    for idx in range(2,13):
        if jja_mo[i] == idx and jja_yr[i] % 4 != 0:
            uwnd = np.mean(np.array(uwnd_filedata.variables["uwnd"][motime[idx-1][0]:motime[idx-1][1],9,:,:]),0)
        elif jja_mo[i] == idx and jja_yr[i] % 4 == 0 and jja_yr[i] != 2000:
            uwnd = np.mean(np.array(uwnd_filedata.variables["uwnd"][motime[idx-1][0]:motime[idx-1][1]+1,9,:,:]),0)

    jja_avg_u[:,:,i] = uwnd

jja_avg_u = np.mean(jja_avg_u,2)





djf_yr = djf.year
djf_mo = djf.mo
djf_avg_u = np.zeros([73,144,len(djf_yr)])

for i in range(len(djf_yr)):
    uwnd_file = "E:/CSL/NCEP/reanalysis data/uwnd."+str(int(djf_yr[i]))+".nc"
    uwnd_filedata = nc.Dataset(uwnd_file)
    uwnd_filedata.variables.keys()

    lon = uwnd_filedata.variables["lon"][:]
    lat = uwnd_filedata.variables["lat"][:]


    motime=[[1,31],[32,59],[60,90],[91,120],[121,151],[152,181],[182,212],[213,243]\
        ,[244,273],[274,304],[305,334],[335,365]]

    if djf_mo[i] == 1:
        uwnd = np.mean(np.array(uwnd_filedata.variables["uwnd"][motime[0][0]:motime[0][1],9,:,:]),0)

    for idx in range(2,13):
        if djf_mo[i] == idx and djf_yr[i] % 4 != 0:
            uwnd = np.mean(np.array(uwnd_filedata.variables["uwnd"][motime[idx-1][0]:motime[idx-1][1],9,:,:]),0)
        elif djf_mo[i] == idx and djf_yr[i] % 4 == 0 and djf_yr[i] != 2000:
            uwnd = np.mean(np.array(uwnd_filedata.variables["uwnd"][motime[idx-1][0]:motime[idx-1][1]+1,9,:,:]),0)

    djf_avg_u[:,:,i] = uwnd

djf_avg_u = np.mean(djf_avg_u,2)


##################################
#plot
##################################

fig = plt.figure()

ax = fig.add_subplot(1,2,1)
map = Basemap(
    projection="merc",
    llcrnrlon=100,
    llcrnrlat=10,
    urcrnrlon=360,
    urcrnrlat=60,
    resolution="h",
)
lons, lats = np.meshgrid(lon, lat)
x, y = map(lons, lats)
# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color='w',alpha=0.01)
map.drawcoastlines()
map.contourf(x, y, jja_avg_u,range(-20,100,10),cmap="hot_r")
# draw lat lon label on map
map.drawparallels(np.arange(int(10), int(60), 10), labels=[1, 0, 0, 0],fontsize=7)
map.drawmeridians(np.arange(int(100), int(360), 50), labels=[0, 0, 0, 1],fontsize=7)

plt.title("jja_avg_u",fontsize=7,fontweight='bold')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=7)




ax = fig.add_subplot(1,2,2)
map = Basemap(
    projection="merc",
    llcrnrlon=100,
    llcrnrlat=10,
    urcrnrlon=360,
    urcrnrlat=60,
    resolution="h",
)
# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color='w',alpha=0.01)
map.drawcoastlines()
map.contourf(x, y, djf_avg_u,range(-20,100,10), cmap="hot_r")

# draw lat lon label on map
map.drawparallels(np.arange(int(10), int(60), 10), labels=[1, 0, 0, 0],fontsize=7)
map.drawmeridians(np.arange(int(100), int(360), 50), labels=[0, 0, 0, 1],fontsize=7)

plt.title("djf_avg_u",fontsize=7,fontweight='bold')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=7)

fig.tight_layout()

plt.savefig("E:/CSL/visual studio/SourceTree/jet stream analysis/fig3.png", dpi=600)
plt.close()
# plt.show()


################################################
# AOI  양 , 음 상위10% , (전체 , 태풍있는 달) 평균 해서 u 그리기
# 그때의 태풍 평균 발생 위치; LMI 위치; 온대저기압으로 죽는 지점 표시   
#############################################
# len(neg_ao_Data)*0.1 ; 상위10% 24개

percentile10_neg = neg_ao_Data.sort_values(by='AO_idx',ascending=True)[0:24]
percentile10_pos = pos_ao_Data.sort_values(by='AO_idx',ascending=False)[0:24]







  