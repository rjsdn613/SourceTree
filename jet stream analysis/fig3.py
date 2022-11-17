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



#####################################################
# JJA , DJF 평균 AOI 값과 제트기류
jja = pd.DataFrame()
djf = pd.DataFrame()
for i in range(len(TCseason_ao)):
    if TCseason_ao.mo[i] == 6 or TCseason_ao.mo[i] == 7 or TCseason_ao.mo[i] == 8:
        jja=jja.append(TCseason_ao.loc[i], ignore_index=True)
    elif TCseason_ao.mo[i] == 11 or TCseason_ao.mo[i] == 12 or TCseason_ao.mo[i] == 1:
        djf=djf.append(TCseason_ao.loc[i], ignore_index=True)    

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
    projection="mill",
    lon_0=180,
    resolution="l",
)
lons, lats = np.meshgrid(lon, lat)
x, y = map(lons, lats)
# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color='w',alpha=0.01)
map.drawcoastlines()
map.contourf(x, y, jja_avg_u,range(-20,100,10),cmap="hot_r")


plt.title("jja_avg_u",fontsize=7,fontweight='bold')

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=7)




ax = fig.add_subplot(1,2,2)
map = Basemap(
    projection="mill",
    lon_0=180,
    resolution="l",
)
# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color='w',alpha=0.01)
map.drawcoastlines()
map.contourf(x, y, djf_avg_u,range(-20,100,10), cmap="hot_r")


plt.title("djf_avg_u",fontsize=7,fontweight='bold')
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
cbar.ax.tick_params(labelsize=7)

fig.tight_layout()

plt.savefig("E:/CSL/visual studio/SourceTree/jet stream analysis/fig3_mill.png", dpi=600)
plt.close()
# plt.show()


################################################
# fig1 전구로 확대해서 그리기
# m = Basemap(projection='robin',lon_0=180,resolution='h')
################################################

################################################
# AOI  양 , 음 상위10% , 태풍있는 달만 평균 해서 u 그리기
# 그때의 태풍 평균 발생 위치; LMI 위치; 온대저기압으로 죽는 지점 표시   
################################################


















