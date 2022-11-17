## Library
import netCDF4 as nc
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from numpy import linspace
from numpy import meshgrid
from mpl_toolkits.axes_grid1 import make_axes_locatable  # divider = make_axes_locatable(gca); LIB

'''
볼라벤 중위도 3일 이동평균 = 2012.8.26 ~ 2012.8.28
솔릭 중위도 3일 이동평균 = 2018.8.21 ~ 2018.8.23
lon lat time = 1440 720 365
sst(time lat lon)
8월 1일 julian day는 214
8월 26일은 239
8월 21일은 234
'''

bolaven = "E:/CSL/visual studio/SourceTree/sst.day.mean.2012.nc"
b_dataset = nc.Dataset(bolaven)
b_dataset.variables.keys()


lon = b_dataset.variables["lon"][:]
lat = b_dataset.variables["lat"][:]
b_sst = b_dataset.variables["sst"][239:242,:,:]  # time(365) lat(720) lon(1440)
b_sst = np.array(b_sst)


soulik = "E:/CSL/visual studio/SourceTree/sst.day.mean.2018.nc"
s_dataset = nc.Dataset(soulik)
s_dataset.variables.keys()
s_sst = s_dataset.variables["sst"][234:237,:,:]  # time(365) lat(720) lon(1440)
s_sst = np.array(s_sst)


Mean_Bsst = np.mean(b_sst, axis=0)
Mean_Ssst = np.mean(s_sst, axis=0)



############################################################################################
# plot
############################################################################################

fig = plt.figure(figsize=(10, 4))


ax = fig.add_subplot(1,2,1)

map = Basemap(
    projection="merc",
    lat_0=40,
    lon_0=135,
    llcrnrlon=100,
    llcrnrlat=20,
    urcrnrlon=150,
    urcrnrlat=50,
    resolution="h",
)
llons, llats = np.meshgrid(lon, lat)
x, y = map(llons, llats)

# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color="grey", lake_color="aqua")
map.drawcoastlines()
map.contourf(x, y, Mean_Bsst[ :, :],range(14, 32, 2),cmap="hot_r")

# draw lat lon label on map
map.drawparallels(np.arange(int(20), int(50), 10), labels=[1, 0, 0, 0])
map.drawmeridians(np.arange(int(110), int(160), 10), labels=[0, 0, 0, 1])

plt.title("Bolaven (T1215)",fontsize=15,fontweight='bold')

# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
"""
orientation (horizontal or vertical)
fraction (default: 0.15. colorbar가 차지하는 영역의 비율)
aspect (default: 20. colorbar의 긴 변 : 짧은 변 비율)
"""




ax = fig.add_subplot(1,2,2)

map = Basemap(
    projection="merc",
    lat_0=40,
    lon_0=135,
    llcrnrlon=100,
    llcrnrlat=20,
    urcrnrlon=150,
    urcrnrlat=50,
    resolution="h",
)
llons, llats = np.meshgrid(lon, lat)
x, y = map(llons, llats)

# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color="grey", lake_color="aqua")
map.drawcoastlines()
map.contourf(x, y, Mean_Ssst[ :, :],range(14, 32, 2),cmap="hot_r")

# draw lat lon label on map
map.drawparallels(np.arange(int(20), int(50), 10), labels=[1, 0, 0, 0])
map.drawmeridians(np.arange(int(110), int(160), 10), labels=[0, 0, 0, 1])


plt.title("Soulik (T1819)",fontsize=15,fontweight='bold')
# create an axes on the right side of ax. The width of cax will be 5%
# of ax and the padding between cax and ax will be fixed at 0.05 inch.
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(orientation="vertical", aspect=20, fraction=0.15, cax=cax)
"""
orientation (horizontal or vertical)
fraction (default: 0.15. colorbar가 차지하는 영역의 비율)
aspect (default: 20. colorbar의 긴 변 : 짧은 변 비율)
"""


plt.suptitle("3days Moving Average SST when 30°N",fontsize=20,fontweight='bold')

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.85, hspace=0.4, wspace=0.4)
plt.savefig("E:/CSL/visual studio/SourceTree/bolaven_soulik_sst.png")
plt.close()

# plt.show()

