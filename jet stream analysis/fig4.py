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
import TC_symbol
##################################################################

ao_data = pd.read_csv('E:/CSL/AO_index2.txt', sep = "\s+")
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


################################################
# AOI  양 , 음 상위10% , (전체 , 태풍있는 달) 평균 해서 u 그리기
# 그때의 태풍 평균 발생 위치; LMI 위치; 온대저기압으로 죽는 지점 표시   
#############################################
# len(TCseason_pos_ao)*0.1 ; 6.1개 , len(TCseason_neg_ao)*0.1 ; 5.9개 

percentile10_neg = TCseason_neg_ao.sort_values(by='AO_idx',ascending=True)[0:6]
percentile10_pos = TCseason_pos_ao.sort_values(by='AO_idx',ascending=False)[0:6]
percentile10_neg=percentile10_neg.reset_index()
percentile10_pos=percentile10_pos.reset_index()

##########################################################################################
pyr = percentile10_pos.yr
pmo = percentile10_pos.mo
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

positive_avg_u = np.mean(p_avg_u,2)



### 변수 초기화
del(uwnd_filedata)
del(uwnd_file)
###


##########################################################################################

##########################################################################################
nyr = percentile10_neg.yr
nmo = percentile10_neg.mo
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

negative_avg_u = np.mean(n_avg_u,2)



### 변수 초기화
del(uwnd_filedata)
del(uwnd_file)
###


####################################################################################
#태풍이 있었던 달 평균에는 태풍의 발생위치 , LMI 위치, 온대저기압으로 소멸된 위치의 평균을 나타낼 것.
####################################################################################
a=percentile10_neg.TC_number
a = a.str.replace(pat=r' ', repl=r'', regex=True)
a = a.str.replace(pat=r'[', repl=r'', regex=True)
a = a.str.replace(pat=r']', repl=r'', regex=True)
b = a.str.split(',').tolist()
neg_tcnum = sum(b,[]) # 이중리스트 풀기
neg_tcnum = [int (i) for i in neg_tcnum] # 리스트를 숫자형으로


start_lat = []
start_lon = []
LMI_lat = []
LMI_lon = []
end_lat = []
end_lon = []
# track_lat = []
# track_lon = []
for i in range(len(neg_tcnum)): 

    tsidx = tc2.get_tc_grade(neg_tcnum[i]).index(3) # 처음으로 TS 등급되는 지점의 인덱스 찾기(발생위치 찾기)
    start_lat.append( tc2.get_tc_lat(neg_tcnum[i])[tsidx] ) # 태풍번호별 발생 위도
    start_lon.append( tc2.get_tc_lon(neg_tcnum[i])[tsidx] ) # 태풍번호별 발생 경도

    lmiidx = tc2.get_tc_wind(neg_tcnum[i]).index(max(tc2.get_tc_wind(neg_tcnum[i]))) # 태풍번호별 LMI index
    LMI_lat.append( tc2.get_tc_lat(neg_tcnum[i])[lmiidx] ) # 태풍번호별 LMI 위도
    LMI_lon.append( tc2.get_tc_lon(neg_tcnum[i])[lmiidx] ) # 태풍번호별 LMI 경도
    
                            #    2 : Tropical Depression (TD) (취급안함, 윗등급에서 내려오면 소멸로 취급)
                            #    3 : Tropical Storm (TS) ; (태풍의 발생위치)
                            #    4 : Severe Tropical Storm (STS)
                            #    5 : Typhoon (TY)
                            #    6 : Extra-tropical Cyclone (L) (3,4,5 에서 6으로 되면 소멸로 취급)
                            #    7 : Just entering into the responsible area of
                            #        RSMC Tokyo-Typhoon Center
                            #    8 : Not used
                            #    9 : Tropical Cyclone of TS intensity or higher


    # 0. 발생이후 2, 6 grade 둘다 없는 경우(간혹 존재, 3으로 끝남)
    if 2 not in tc2.get_tc_grade(neg_tcnum[i])[tsidx::] and 6 not in tc2.get_tc_grade(neg_tcnum[i])[tsidx::]:
        trigger = list(np.where( np.array(tc2.get_tc_grade(neg_tcnum[i])) == 3 )[0] > tsidx).index(True) 
        endidx  = np.where( np.array(tc2.get_tc_grade(neg_tcnum[i])) == 3 )[0][trigger] # 발생(TS)이후 처음으로 6 grade 되는 인덱스찾기        
    # 1. 발생이후 6 grade만 있는 경우
    elif 2 not in tc2.get_tc_grade(neg_tcnum[i])[tsidx::]:  # 발생이후에 2 grade가 없으면 6이 있다는 것.
        trigger = list(np.where( np.array(tc2.get_tc_grade(neg_tcnum[i])) == 6 )[0] > tsidx).index(True) 
        endidx  = np.where( np.array(tc2.get_tc_grade(neg_tcnum[i])) == 6 )[0][trigger] # 발생(TS)이후 처음으로 6 grade 되는 인덱스찾기

    # 2. 발생이후 2 grade만 있는 경우
    elif 6 not in tc2.get_tc_grade(neg_tcnum[i])[tsidx::]:  # 발생이후에 6 grade가 없으면 2이 있다는 것.
        trigger = list(np.where( np.array(tc2.get_tc_grade(neg_tcnum[i])) == 2 )[0] > tsidx).index(True) 
        endidx  = np.where( np.array(tc2.get_tc_grade(neg_tcnum[i])) == 2 )[0][trigger] # 발생(TS)이후 처음으로 2 grade 되는 인덱스찾기

    # 3. 발생이후 6,2 grade 둘다 있는 경우 
    elif 2 in tc2.get_tc_grade(neg_tcnum[i])[tsidx::] or 6 in tc2.get_tc_grade(neg_tcnum[i])[tsidx::]:

        trigger1 = list( np.where( np.array(tc2.get_tc_grade(neg_tcnum[i])) == 2 )[0] > tsidx ).index(True) 
        endidx1  = np.where( np.array(tc2.get_tc_grade(neg_tcnum[i])) == 2 )[0][trigger1] # 발생(TS)이후 처음으로 2 grade 되는 인덱스찾기
        trigger2 = list( np.where( np.array(tc2.get_tc_grade(neg_tcnum[i])) == 6 )[0] > tsidx ).index(True) 
        endidx2  = np.where( np.array(tc2.get_tc_grade(neg_tcnum[i])) == 6 )[0][trigger2] # 발생(TS)이후 처음으로 2 grade 되는 인덱스찾기
        endidx = min(endidx1,endidx2)

    end_lat.append( tc2.get_tc_lat(neg_tcnum[i])[endidx] )  # 태풍번호별 소멸 위도
    end_lon.append( tc2.get_tc_lon(neg_tcnum[i])[endidx] )  # 태풍번호별 소멸 경도



neg_avg_start_lat = np.mean(start_lat)
neg_avg_start_lon = np.mean(start_lon)
neg_avg_LMI_lat = np.mean(LMI_lat)
neg_avg_LMI_lon = np.mean(LMI_lon)
neg_avg_end_lat = np.mean(end_lat)
neg_avg_end_lon = np.mean(end_lon)

#########################################################

a=percentile10_pos.TC_number
a = a.str.replace(pat=r' ', repl=r'', regex=True)
a = a.str.replace(pat=r'[', repl=r'', regex=True)
a = a.str.replace(pat=r']', repl=r'', regex=True)
b = a.str.split(',').tolist()
pos_tcnum = sum(b,[]) # 이중리스트 풀기
pos_tcnum = [int (i) for i in pos_tcnum] # 리스트를 숫자형으로


start_lat = []
start_lon = []
LMI_lat = []
LMI_lon = []
end_lat = []
end_lon = []
for i in range(len(pos_tcnum)): 
    tsidx = tc2.get_tc_grade(pos_tcnum[i]).index(3) # 처음으로 TS 등급되는 지점의 인덱스 찾기(발생위치 찾기)
    start_lat.append( tc2.get_tc_lat(pos_tcnum[i])[tsidx] ) # 태풍번호별 발생 위도
    start_lon.append( tc2.get_tc_lon(pos_tcnum[i])[tsidx] ) # 태풍번호별 발생 경도

    lmiidx = tc2.get_tc_wind(pos_tcnum[i]).index(max(tc2.get_tc_wind(pos_tcnum[i]))) # 태풍번호별 LMI index
    LMI_lat.append( tc2.get_tc_lat(pos_tcnum[i])[lmiidx] ) # 태풍번호별 LMI 위도
    LMI_lon.append( tc2.get_tc_lon(pos_tcnum[i])[lmiidx] ) # 태풍번호별 LMI 경도
    
                            #    2 : Tropical Depression (TD) (취급안함, 윗등급에서 내려오면 소멸로 취급)
                            #    3 : Tropical Storm (TS) ; (태풍의 발생위치)
                            #    4 : Severe Tropical Storm (STS)
                            #    5 : Typhoon (TY)
                            #    6 : Extra-tropical Cyclone (L) (3,4,5 에서 6으로 되면 소멸로 취급)
                            #    7 : Just entering into the responsible area of
                            #        RSMC Tokyo-Typhoon Center
                            #    8 : Not used
                            #    9 : Tropical Cyclone of TS intensity or higher


    # 0. 발생이후 2, 6 grade 둘다 없는 경우(간혹 존재, 3으로 끝남)
    if 2 not in tc2.get_tc_grade(pos_tcnum[i])[tsidx::] and 6 not in tc2.get_tc_grade(pos_tcnum[i])[tsidx::]:
        trigger = list(np.where( np.array(tc2.get_tc_grade(pos_tcnum[i])) == 3 )[0] > tsidx).index(True) 
        endidx  = np.where( np.array(tc2.get_tc_grade(pos_tcnum[i])) == 3 )[0][trigger] # 발생(TS)이후 처음으로 6 grade 되는 인덱스찾기        
    # 1. 발생이후 6 grade만 있는 경우
    elif 2 not in tc2.get_tc_grade(pos_tcnum[i])[tsidx::]:  # 발생이후에 2 grade가 없으면 6이 있다는 것.
        trigger = list(np.where( np.array(tc2.get_tc_grade(pos_tcnum[i])) == 6 )[0] > tsidx).index(True) 
        endidx  = np.where( np.array(tc2.get_tc_grade(pos_tcnum[i])) == 6 )[0][trigger] # 발생(TS)이후 처음으로 6 grade 되는 인덱스찾기

    # 2. 발생이후 2 grade만 있는 경우
    elif 6 not in tc2.get_tc_grade(pos_tcnum[i])[tsidx::]:  # 발생이후에 6 grade가 없으면 2이 있다는 것.
        trigger = list(np.where( np.array(tc2.get_tc_grade(pos_tcnum[i])) == 2 )[0] > tsidx).index(True) 
        endidx  = np.where( np.array(tc2.get_tc_grade(pos_tcnum[i])) == 2 )[0][trigger] # 발생(TS)이후 처음으로 2 grade 되는 인덱스찾기

    # 3. 발생이후 6,2 grade 둘다 있는 경우 
    elif 2 in tc2.get_tc_grade(pos_tcnum[i])[tsidx::] or 6 in tc2.get_tc_grade(pos_tcnum[i])[tsidx::]:

        trigger1 = list( np.where( np.array(tc2.get_tc_grade(pos_tcnum[i])) == 2 )[0] > tsidx ).index(True) 
        endidx1  = np.where( np.array(tc2.get_tc_grade(pos_tcnum[i])) == 2 )[0][trigger1] # 발생(TS)이후 처음으로 2 grade 되는 인덱스찾기
        trigger2 = list( np.where( np.array(tc2.get_tc_grade(pos_tcnum[i])) == 6 )[0] > tsidx ).index(True) 
        endidx2  = np.where( np.array(tc2.get_tc_grade(pos_tcnum[i])) == 6 )[0][trigger2] # 발생(TS)이후 처음으로 2 grade 되는 인덱스찾기
        endidx = min(endidx1,endidx2)

    end_lat.append( tc2.get_tc_lat(pos_tcnum[i])[endidx] )  # 태풍번호별 소멸 위도
    end_lon.append( tc2.get_tc_lon(pos_tcnum[i])[endidx] )  # 태풍번호별 소멸 경도



pos_avg_start_lat = np.mean(start_lat)
pos_avg_start_lon = np.mean(start_lon)
pos_avg_LMI_lat = np.mean(LMI_lat)
pos_avg_LMI_lon = np.mean(LMI_lon)
pos_avg_end_lat = np.mean(end_lat)
pos_avg_end_lon = np.mean(end_lon)


###########################################################################################
#plot
###########################################################################################
hurricane = TC_symbol.get_hurricane() # 커스텀 태풍 심볼

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
f1=map.contourf(x, y, positive_avg_u, range(-10,90,10), cmap="hot_r")

# Map (lon, lat) to (x, y) for plotting
x1, y1 = map(pos_avg_start_lon, pos_avg_start_lat)
plt.scatter(x1, y1, marker=hurricane, s=40 , edgecolors="crimson", facecolors='none', linewidth=1)
x2, y2 = map(pos_avg_LMI_lon, pos_avg_LMI_lat)
plt.scatter(x2, y2, marker=hurricane, s=40 , edgecolors="green", facecolors='none', linewidth=1)
x3, y3 = map(pos_avg_end_lon, pos_avg_end_lat)
plt.scatter(x3, y3, marker=hurricane, s=40 , edgecolors="blue", facecolors='none', linewidth=1)


# draw lat lon label on map
map.drawparallels(np.arange(int(0), int(80), 20), labels=[1, 0, 0, 0],fontsize=5)
map.drawmeridians(np.arange(int(60), int(360), 60), labels=[0, 0, 0, 1],fontsize=5)

plt.title("positive_avg_u (percentile 10)",fontsize=7,fontweight='bold')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(f1,orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=5)




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


f2=map1.contourf(x, y, negative_avg_u,range(-10,90,10), cmap="hot_r")


# Map (lon, lat) to (x, y) for plotting
x1, y1 = map1(neg_avg_start_lon, neg_avg_start_lat)
plt.scatter(x1, y1, marker=hurricane, s=40 , edgecolors="crimson", facecolors='none', linewidth=1)
x2, y2 = map1(neg_avg_LMI_lon, neg_avg_LMI_lat)
plt.scatter(x2, y2, marker=hurricane, s=40 , edgecolors="green", facecolors='none', linewidth=1)
x3, y3 = map1(neg_avg_end_lon, neg_avg_end_lat)
plt.scatter(x3, y3, marker=hurricane, s=40 , edgecolors="blue", facecolors='none', linewidth=1)


# draw lat lon label on map
map1.drawparallels(np.arange(int(0), int(80), 20), labels=[1, 0, 0, 0],fontsize=5)
map1.drawmeridians(np.arange(int(60), int(360), 60), labels=[0, 0, 0, 1],fontsize=5)


plt.title("negative_avg_u (percentile 10)",fontsize=7,fontweight='bold')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(f2,orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=5)




plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, hspace=0.1, wspace=0.3)
# plt.show()



plt.savefig("E:/CSL/visual studio/SourceTree/jet stream analysis/fig4_Mid.png", dpi=600)
plt.close()
