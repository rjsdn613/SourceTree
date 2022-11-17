## Library
import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimationimport
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from numpy import linspace
from numpy import meshgrid
from mpl_toolkits.axes_grid1 import make_axes_locatable  # divider = make_axes_locatable(gca); LIB
import read_bst_file3 as tc_rsmc
import read_HURDAT as tc_nhc
from collections import Counter
from random import *
from matplotlib import gridspec
import netCDF4 as nc
import TC_symbol

'''
def animate(i):
    x = np.linspace(0, 4, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig, animate, frames=200, interval=200)

plt.show()
anim.save('test.gif', writer='imagemagick')
'''

for i in range(1,1029):                             # rsmc 1981년 ~ 2020년 까지의 범위
    n = len(tc_rsmc.get_bst_NWP(i)) 
    for j in range(n):
        yr = tc_rsmc.get_bst_NWP(i)['date'][j][0:2] # 해당 태풍의 시작 년도
        if yr > 50:
            yr=yr+1900
        elif yr < 50:
            yr=yr+2000
        mo = int(tc_rsmc.get_bst_NWP(i)['date'][j][2:4]) # 해당 태풍의 시작 월
        dy = int(tc_rsmc.get_bst_NWP(i)['date'][j][4:6]) # 해당 태풍의 시작 일
        hr = int(tc_rsmc.get_bst_NWP(i)['date'][j][6:8]) # 해당 태풍의 시작 시각

        # hr을 index로 만들기
        if hr == 0:
            ihr=1
        elif hr == 6:
            ihr=2
        elif hr == 12:
            ihr=3            
        elif hr == 18:
            ihr=4
        # dy을 index로 만들기
        ldy = list(range(1,32))
        kdy = list(range(0,4*31,4))
        for k in range(31):
            if dy == ldy[k]:
                idy=kdy[k]
                
        #julian day로 변환 (윤년 구분)
        if yr % 4 != 0 or yr == 2000:          #평년
            jmo=np.multiply([0,31,59,90,120,151,181,212,243,273,304,334],[4])
        elif yr[i] % 4 == 0 and yr != 2000:    #윤년
            jmo=np.multiply([0,31,59+1,90+1,120+1,151+1,181+1,212+1,243+1,273+1,304+1,334+1],[4])

        if mo == 6:
            jdy = jmo[5] + idy + ihr
        elif mo == 7:
            jdy = jmo[6] + idy + ihr
        elif mo == 8:
            jdy = jmo[7] + idy + ihr
        elif mo == 9:
            jdy = jmo[8] + idy + ihr
        elif mo == 10:
            jdy = jmo[9] + idy + ihr
        else:
            print("Number",i,"TC is not in TC season")
        # elif mo == 1:
        #     jdy = jmo[0]+dy+jhr
        # elif mo == 2:
        #     jdy = jmo[1]+dy+jhr
        # elif mo == 3:
        #     jdy = jmo[2]+dy+jhr
        # elif mo == 4:
        #     jdy = jmo[3]+dy+jhr
        # elif mo == 5:
        #     jdy = jmo[4]+dy+jhr
        # elif mo == 11:
        #     jdy = jmo[10]+dy+jhr
        # elif mo == 12:
        #     jdy = jmo[11]+dy+jhr



        uwnd_file = "E:/CSL/NCEP/reanalysis data/uwnd."+yr+".nc"
        uwnd_filedata = nc.Dataset(uwnd_file)
        uwnd_filedata.variables.keys()
        lon = uwnd_filedata.variables["lon"][:]
        lat = uwnd_filedata.variables["lat"][:]
        uwnd = uwnd_filedata.variables["uwnd"][jdy,9,:,:]  # time level lat lon
        uwnd_filedata.close()
        


 







# tc_rsmc.get_bst_NWP(1:1028)
# tc_nhc.get_HURDAT_NEP_and_NCP(374: 1169)
# tc_nhc.get_HURDAT_ATL(1294 : 1924)

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


ax = fig.add_subplot(1,2,2)
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