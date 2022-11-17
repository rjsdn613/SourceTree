from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as crs
from cartopy.feature import NaturalEarthFeature

from wrf import (getvar, interplevel, to_np, latlon_coords, get_cartopy,
                 cartopy_xlim, cartopy_ylim)

# Open the NetCDF file
ncfile = Dataset("E:/CSL/visual studio/SourceTree/WRF/x1.3/wrfout_d01_2016-10-05_00_00_00")

# Extract the pressure, geopotential height, and wind variables
p = getvar(ncfile, "pressure")
z = getvar(ncfile, "z", units="dm")
ua = getvar(ncfile, "ua", units="ms-1")
va = getvar(ncfile, "va", units="ms-1")
wspd = getvar(ncfile, "wspd_wdir", units="ms-1")[0,:]

# Interpolate geopotential height, u, and v winds to 500 hPa
ht_500 = interplevel(z, p, 250)
u_500 = interplevel(ua, p, 250)
v_500 = interplevel(va, p, 250)
wspd_500 = interplevel(wspd, p, 250)

# Get the lat/lon coordinates
lats, lons = latlon_coords(ht_500)

# Get the map projection information
cart_proj = get_cartopy(ht_500)

# Create the figure
fig = plt.figure(figsize=(12,9))
ax = plt.axes(projection=cart_proj)

# Download and add the states and coastlines
states = NaturalEarthFeature(category="cultural", scale="50m",
                             facecolor="none",
                             name="admin_1_states_provinces_shp")
ax.add_feature(states, linewidth=0.5, edgecolor="black")
ax.coastlines('50m', linewidth=0.8)


# Add the wind speed contours
levels = [-70,-60,-50,-40,-30,-20,-10,0,10,20,30,40,50,60,70]
wspd_contours = plt.contourf(to_np(lons), to_np(lats), to_np(u_500),
                             levels=levels,
                             cmap=get_cmap("RdBu_r"),
                             transform=crs.PlateCarree())
plt.colorbar(wspd_contours, ax=ax, orientation="horizontal", pad=.05)


# Set the map bounds
ax.set_xlim(cartopy_xlim(ht_500))
ax.set_ylim(cartopy_ylim(ht_500))

ax.gridlines()

plt.title("500 MB Height (dm), Wind Speed (kt), Barbs (kt)")

plt.show()