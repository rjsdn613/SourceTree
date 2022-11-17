import os, sys
import pandas as pd
import numpy as np
from numpy import ma
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap
import netCDF4 as nc
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cartopy.crs as ccrs
import cartopy.feature as cfeature

file = "E:/CSL/visual studio/SourceTree/sst.mnmean.nc"
filedata = nc.Dataset(file)
filedata.variables.keys()

lon = np.array(filedata.variables["lon"][:])
lat = np.array(filedata.variables["lat"][:])
time = np.array(filedata.variables["time"][:])
sst = np.array(filedata.variables["sst"][:,:,:]) # time(463) lat(180) lon(360) 


# 데이터 크니까 공간(관측소) , 시간(변수) 을 바꿔서 , 열 평균이 아닌 행평균을 빼서 계산
X = np.reshape(sst, ( len(time),  len(lat)*len(lon) ))
X = X - np.reshape(np.nanmean(X,axis=1) ,(463,1))
cov = X@X.T  / 3600


# calculate eigenvalues and eigenvectors
D,V = np.linalg.eig(cov)
idx = D.argsort()[::-1] # 내림차순정렬 , D.argsort() = 오름차순정렬
D = D[idx]
V = V[:,idx]
print('Eigenvalues\n',D)
print('Eigenvectors\n',V)

pc  = V.T @ X

'''
- row vector로 변환하려면:  array_1d.reshape((1, -1))  # -1 은 해당 axis의 size를 자동 결정하라는 뜻 

- column vector로 변환하려면; array_1d.reshape((-1, 1))
'''

eof1 = np.reshape(pc[0,:],(-1,1)) @ np.reshape(V[:,0],(1,-1))
eof2 = np.reshape(pc[1,:],(-1,1)) @ np.reshape(V[:,1],(1,-1))
eof3 = np.reshape(pc[2,:],(-1,1)) @ np.reshape(V[:,2],(1,-1))


sst_eof1 = np.reshape(eof1, (len(lat) , len(lon),len(time)))
sst_eof2 = np.reshape(eof2, (len(lat) , len(lon),len(time)))
sst_eof3 = np.reshape(eof3, (len(lat) , len(lon),len(time)))


#%%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(2,1,1)

map = Basemap(
    projection="merc",
    llcrnrlon=90,
    llcrnrlat=-20,
    urcrnrlon=300,
    urcrnrlat=60,
    resolution="h",
)
llons, llats = np.meshgrid(lon, lat)
x, y = map(llons, llats)
# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color="grey", lake_color="aqua")
map.drawcoastlines()
map.contourf(x,y,sst_eof1[:,:,0],cmap="Reds")
map.colorbar()

ax = fig.add_subplot(2,1,2)
plt.plot(V[0,:])


plt.suptitle("EOF1",fontsize=20,fontweight='bold')

plt.savefig("E:/CSL/visual studio/SourceTree/EOF1.png")
plt.close()

#%%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(2,1,1)

map = Basemap(
    projection="merc",
    llcrnrlon=90,
    llcrnrlat=-20,
    urcrnrlon=300,
    urcrnrlat=60,
    resolution="h",
)
llons, llats = np.meshgrid(lon, lat)
x, y = map(llons, llats)
# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color="grey", lake_color="aqua")
map.drawcoastlines()
map.contourf(x,y,sst_eof2[:,:,0],cmap="Reds")
map.colorbar()

ax = fig.add_subplot(2,1,2)
plt.plot(V[1,:])

plt.suptitle("EOF2",fontsize=20,fontweight='bold')

plt.savefig("E:/CSL/visual studio/SourceTree/EOF2.png")
plt.close()

#%%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(2,1,1)

map = Basemap(
    projection="merc",
    llcrnrlon=90,
    llcrnrlat=-20,
    urcrnrlon=300,
    urcrnrlat=60,
    resolution="h",
)
llons, llats = np.meshgrid(lon, lat)
x, y = map(llons, llats)
# map.drawmapboundary(fill_color='aqua').
map.fillcontinents(color="grey", lake_color="aqua")
map.drawcoastlines()
map.contourf(x,y,sst_eof3[:,:,0],cmap="Reds")
map.colorbar()

ax = fig.add_subplot(2,1,2)
plt.plot(V[2,:])

plt.suptitle("EOF3",fontsize=20,fontweight='bold')

plt.savefig("E:/CSL/visual studio/SourceTree/EOF3.png")
plt.close()




tot = sum(D[0:3])
var_exp = [(i / tot)*100 for i in sorted(D[0:3], reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print('var_exp\n',var_exp)
print('cum_var_exp\n',cum_var_exp)



























# # Plot the leading EOF expressed as correlation in the Pacific domain.
# clevs = np.linspace(-1, 1, 11)
# ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=190))
# fill = ax.contourf(lon, lat, sst_eof1[:,:,0], clevs,
#                    transform=ccrs.PlateCarree(), cmap=plt.cm.RdBu_r)
# ax.add_feature(cfeature.LAND, facecolor='w', edgecolor='k')
# cb = plt.colorbar(fill, orientation='horizontal')
# cb.set_label('correlation coefficient', fontsize=12)
# plt.title('EOF1 expressed as correlation', fontsize=16)







# pca = PCA(n_components=3)
# principalComponents = pca.fit_transform(x)
# # principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1','pc2','pc3'])

# f, ax = plt.subplots(figsize=(6,6))
# ax.plot(pca.explained_variance_ratio_[0:10]*100)
# ax.plot(pca.explained_variance_ratio_[0:10]*100,'ro')
# ipc = np.where(pca.explained_variance_ratio_.cumsum() >= 0.70)[0][0]






