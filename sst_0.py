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

filename = "E:/CSL/sst.day.mean.2019.nc"
dataset = nc.Dataset(filename)
dataset.variables.keys()

lon = dataset.variables[u"lon"][:]
lat = dataset.variables[u"lat"][:]
sst = dataset.variables[u"sst"][:]  # time(365) lat(720) lon(1440)
time = dataset.variables[u"time"][:]


plt.figure()
ax = plt.gca()

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
map.contourf(x, y, sst[1, :, :], range(-6, 33, 3), cmap="coolwarm")

# draw lat lon label on map
map.drawparallels(np.arange(int(20), int(50), 10), labels=[1, 0, 0, 0])
map.drawmeridians(np.arange(int(110), int(160), 10), labels=[0, 0, 0, 1])

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

