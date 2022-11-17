## Library
import matplotlib.pyplot as plt
from matplotlib import animation, rc
import pandas as pd
import numpy as np
from IPython.display import display
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from numpy import linspace
from numpy import meshgrid
from mpl_toolkits.axes_grid1 import make_axes_locatable  # divider = make_axes_locatable(gca); LIB
import read_bst_file3 as RSMC
import read_HURDAT as HURDAT
from collections import Counter
from random import *
from matplotlib import gridspec
import netCDF4 as nc
import TC_symbol


def animate():
    for i in range(1,1029):                             # rsmc 1981년 ~ 2020년 까지의 범위 (1~1028)
        n = len(RSMC.NWP(i)) 
        for j in range(n):
            yr = int(RSMC.NWP(i)['date'][j][0:2]) # 해당 태풍의 시작 년도
            if yr > 50:
                yr=yr+1900
            elif yr < 50:
                yr=yr+2000
            mo = int(RSMC.NWP(i)['date'][j][2:4]) # 해당 태풍의 시작 월
            dy = int(RSMC.NWP(i)['date'][j][4:6]) # 해당 태풍의 시작 일
            hr = int(RSMC.NWP(i)['date'][j][6:8]) # 해당 태풍의 시작 시각

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
            if yr % 4 != 0:          #평년
                jmo=np.multiply([0,31,59,90,120,151,181,212,243,273,304,334],[4])
            elif yr[i] % 4 == 0:    #윤년
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


            uwnd_file = "E:/CSL/NCEP/reanalysis data/uwnd."+str(yr)+".nc"
            uwnd_filedata = nc.Dataset(uwnd_file)
            uwnd_filedata.variables.keys()
            LON = uwnd_filedata.variables["lon"][:]
            LAT = uwnd_filedata.variables["lat"][:]
            uwnd = uwnd_filedata.variables["uwnd"][jdy,9,:,:]  # time level lat lon
            uwnd_filedata.close()
            #상층제트 50kts (약 25m/s) 이상의 풍속대. 


            fig = plt.figure()
            # ax = fig.add_subplot(1,2,1)

            map = Basemap(
                projection="merc",
                llcrnrlon=90,
                llcrnrlat=0,
                urcrnrlon=180,
                urcrnrlat=60,
                resolution="l",
            )

            lons, lats = meshgrid(LON, LAT)
            x, y = map(lons, lats)

            
            map.fillcontinents(color='w',alpha=0.01)
            map.drawcoastlines()

            f1=map.contourf(x, y, uwnd, range(-10,90,10), cmap="RdBu_r")

            # 25m/s 이상 풍속중 중위도 이상만 빨간색으로 표시하기 위함
            ln, lt = meshgrid(LON, LAT[12:33])
            xx, yy = map(ln, lt)        
            f2=map.contour(xx, yy, uwnd[12:33,:], levels=range(25,26), linewidths=1, colors='r')

        # draw lat lon label on map
        map.drawparallels(np.arange(0, 90, 10), labels=[1, 0, 0, 0],fontsize='3')
        map.drawmeridians(np.arange(0, 360, 30), labels=[0, 0, 0, 1],fontsize='3')

        # 태풍 심볼과 정보 입력
        if RSMC.NWP(i)['grade'][j] == str(2):
            edge_color = 'yellow'
            face_color = 'none'
            mk= 'o'
            lw=1
            size = 80
        elif RSMC.NWP(i)['grade'][j] == str(3):
            edge_color = 'yellow' 
            face_color = 'none'
            mk= TC_symbol.get_hurricane() # 커스텀 태풍 심볼
            lw=1
            size = 120
        elif RSMC.NWP(i)['grade'][j] == str(4):
            edge_color = 'orange' 
            face_color = 'none'
            mk= TC_symbol.get_hurricane() 
            size = 120
            lw=1            
        elif RSMC.NWP(i)['grade'][j] == str(5):
            edge_color = 'red' 
            face_color = 'none'
            mk= TC_symbol.get_hurricane()    
            lw=1 
            size = 120
        elif RSMC.NWP(i)['grade'][j] == str(6):
            edge_color = 'yellow'
            face_color = 'yellow'
            mk= r'$\bigotimes$'
            lw=1
            size = 80

        x1, y1 = map(RSMC.NWP(i)['lon'][j], RSMC.NWP(i)['lat'][j]) # 7.5 167.5
        plt.scatter(x1,y1, marker=mk, s=size, edgecolors=edge_color, facecolors=face_color, linewidth=lw)

        font3 = {'family': 'Arial',
        'color':  'black',
        'style': 'italic',
        'weight': 'bold',
        'size': 11}
        box = {'boxstyle': 'round',
        'ec': (1.0, 0.5, 0.5),
        'fc': (1.0, 0.8, 0.8)}
        txt1=plt.text(x1+320000,y1-240000,"{}m/s \n{}hPa".format(RSMC.NWP(i)['spd'][j],
            RSMC.NWP(i)['pres'][j]),fontdict=font3, bbox=box)
        
 




anim = animation.FuncAnimation(fig, animate(), frames=1, interval=200, blit=True)
















# RSMC.NWP(1:1028)
# tc_nhc.get_HURDAT_NEP_and_NCP(374: 1169)
# tc_nhc.get_HURDAT_ATL(1294 : 1924)

