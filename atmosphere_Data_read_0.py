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

hgt_file = "E:/CSL/visual studio/SourceTree/hgt.mon.ltm.nc"
hgt_filedata = nc.Dataset(hgt_file)
hgt_filedata.variables.keys()

lon = hgt_filedata.variables["lon"][:]
lat = hgt_filedata.variables["lat"][:]
hgt = hgt_filedata.variables["hgt"][:]  # time(12) level(17) lat(73) lon(144)
time = hgt_filedata.variables["time"][:]
level = hgt_filedata.variables["level"][:]
'''
level = 1000 925 850 700 600 500 400 300 250 200 150 100 70 50 30 20 10
'''

wspd_file = "E:/CSL/visual studio/SourceTree/wspd.mon.ltm.nc"
wspd_filedata = nc.Dataset(wspd_file)
wspd_filedata.variables.keys()
wspd = wspd_filedata.variables["wspd"][:]  # time(12) level(17) lat(73) lon(144)



uwnd_file = "E:/CSL/visual studio/SourceTree/uwnd.mon.ltm.nc"
uwnd_filedata = nc.Dataset(uwnd_file)
uwnd_filedata.variables.keys()
uwnd = uwnd_filedata.variables["uwnd"][:]  # time(12) level(17) lat(73) lon(144)

vwnd_file = "E:/CSL/visual studio/SourceTree/vwnd.mon.ltm.nc"
vwnd_filedata = nc.Dataset(vwnd_file)
vwnd_filedata.variables.keys()
vwnd = vwnd_filedata.variables["vwnd"][:]  # time(12) level(17) lat(73) lon(144)
###############################################################################

plt.figure()
ax = plt.gca()

map = Basemap(
    projection="merc",
    llcrnrlon=50,
    llcrnrlat=0,
    urcrnrlon=180,
    urcrnrlat=70,
    resolution="h",
)
llons, llats = np.meshgrid(lon, lat)
x, y = map(llons, llats)

# map.drawmapboundary(fill_color='aqua').
# map.fillcontinents(color="grey", lake_color="aqua")
map.drawcoastlines()

# map.contourf(x, y, hgt[8,10, :,:], range(-6, 33, 3), cmap="coolwarm")
map.contourf(x, y, hgt[9,10, :,:], cmap="coolwarm")
map.quiver(x,y,uwnd[9,10, :,:],vwnd[9,10, :,:], angles='xy', pivot='middle')



# draw lat lon label on map
# map.drawparallels(np.arange(int(20), int(50), 10), labels=[1, 0, 0, 0])
# map.drawmeridians(np.arange(int(100), int(150), 10), labels=[0, 0, 0, 1])

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

plt.show()
