
'''
We can clearly see that all three approaches yield the same eigenvectors and eigenvalue pairs:

Eigendecomposition of the covariance matrix after standardizing the data.
Eigendecomposition of the correlation matrix.
Eigendecomposition of the correlation matrix after standardizing the data.

'''

import netCDF4 as nc
import numpy as np
import numpy.ma as ma


from IPython.display import display
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd

a = np.random.rand(3, 3)

# A = a - np.mean(a,axis=0)
A = StandardScaler().fit_transform(a)



cov = np.dot(A , A.T) / 2
np.cov(A)


print(cov)
# calculate eigenvalues and eigenvectors
D,V = np.linalg.eig(cov)
idx = D.argsort()[::-1] # 내림차순정렬 , D.argsort() = 오름차순정렬
D = D[idx]
V = V[:,idx]
print('Eigenvalues\n',D)
print('Eigenvectors\n',V)

# # get the pc1 and pc2
# v1 = -V[:,0] # Ensure that the feature vector direction is consistent with the projection direction calculated later, so add a minus sign
# v2 = V[:,1]
# print('Principal component pcv-1',v1)
# V[:,0] = - V[:,0] 


pc = np.dot(A.T, V)


eof1 = np.dot(np.reshape(V[:,0],(-1,1)) , np.reshape(pc[:,0],(1,3)))
eof2 = np.dot(np.reshape(V[:,1],(-1,1)) , np.reshape(pc[:,1],(1,3)))
eof3 = np.dot(np.reshape(V[:,2],(-1,1)) , np.reshape(pc[:,2],(1,3)))



plt.figure()
raw,=plt.plot(A[:,0],label='raw')
eof1,=plt.plot(eof1[:,0],label='eof1')
eof2,=plt.plot(eof2[:,0],label='eof2')
eof3,=plt.plot(eof3[:,0],label='eof3')
plt.legend([raw,eof1,eof2,eof3],['raw','eof1','eof2','eof3'])

plt.show()

tot = sum(D)
var_exp = [(i / tot)*100 for i in sorted(D, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

print('var_exp\n',var_exp)
print('cum_var_exp\n',cum_var_exp)


######################################################################

























# import netCDF4 as nc
# import pandas as pd
# import numpy as np
# import numpy.ma as ma
# # np.set_printoptions(threshold=np.inf, linewidth=np.inf) #inf = infinity 
# from IPython.display import display
# import matplotlib
# import matplotlib.pyplot as plt


# file = "E:/CSL/visual studio/SourceTree/sst.mnmean.nc"
# filedata = nc.Dataset(file)
# filedata.variables.keys()

# sst = np.array(filedata.variables["sst"][:]) # time(463) lat(180) lon(360) 


# a=-1
# SST = np.zeros([463,180*360])
# for i in range(180):
#     for j in range(360):
    
#         a+=1
#         SST[:,a] = sst[:,i,j]

# '''
# mnsst = np.array(np.mean(SST,axis=0))
# X = SST - mnsst
# cov_sst=  np.dot(X,X.T) /462
# '''


# cov_sst=np.cov(SST.T, rowvar = False)
# eigen_val, eigen_vec = np.linalg.eig(cov_sst) 

# pc = SST.T@eigen_val









# ######################################################################################


# import numpy as np
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()

# def data_vis(d,s=0.5):
#     ax.scatter(d[:,0],d[:,1],s=s)
#     ax.grid()

# def drawArrow1(B,m,c="pc1",):
#     # fc: filling color
#     # ec: edge color
#     if c=='pc1':
#         fc='r'
#         ec='r'
#         s=0.1
#     else:
#         fc='g'
#         ec='g'
#         s=0.1

#     ax.arrow(m[0][0],m[1][0], B[0], B[1],
#              length_includes_head=True,# Increased length includes arrow part
#              head_width=s, head_length=s, fc=fc, ec=ec,label='abc')
#     # Note: The default display range [0,1][0,1], you need to set the graphic range separately to display the arrow
#     ax.set_xticks(np.linspace(0,4,9))
#     ax.set_yticks(np.linspace(0,4,9))
#     ax.set_xlim(0,4)
#     ax.set_ylim(0,4)
#     ax.set_aspect('equal') #x axis y axis is proportional


# # make data    
# data = np.array([[2.5,2.4],[0.5,0.7],[2.2,2.9],[1.9,2.2],[3.1,3.0],
#                  [2.3,2.7],[2.0,1.6],[1.0,1.1],[1.5,1.6],[1.1,0.9]]).T
# m = np.mean(data,axis=1,keepdims=1)
# data_adjust = data-m
# data_adjust2 = data_adjust.T
# # visualize data
# data_vis(data.T,s=10)



# # calcalate conv
# c = np.cov(data_adjust)
# print('Covariance matrix:\n',c)



# # calculate eigenvalues and eigenvectors
# D,V = np.linalg.eig(c)
# idx = D.argsort()[::-1]   
# D = D[idx]
# V = V[:,idx]
# print('Characteristic value\n',D)
# print('Feature vector\n',V)


# #get the pc1 and pc2
# v1 = -V[:,0] # Ensure that the feature vector direction is consistent with the projection direction calculated later, so add a minus sign
# v2 = V[:,1]
# print('Principal component pcv-1',v1)


# # visual eigenvector
# drawArrow1(v2,m,c='pc2')
# drawArrow1(v1,m,c='pc1')


# # calculate the final result
# final = np.dot(data_adjust.T, v1)


# #calculate the final coordinate
# theta = np.arctan(v1[1]/v1[0])
# print('The angle between the principal component pcv-1 and the x axis θ %f degree'%(theta/np.pi*180))
# final_x = (final)*np.cos(theta)+m[0]
# final_y = (final)*np.sin(theta)+m[1]   
# final_xy = np.vstack((final_x,final_y))
# data_vis(final_xy.T,s=10)
# ax.grid()

# #y = k*(x-m[0])+m[1]

# k = np.tan(theta)
# m = m.reshape(2)
# x1, y1 = [0, 4], [-k*m[0]+m[1],k*(4-m[0])+m[1]]
# plt.plot(x1, y1, 'y--',linewidth=0.5)
# plt.show()
