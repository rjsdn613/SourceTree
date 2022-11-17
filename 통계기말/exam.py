# 데이터 만들기
#########################################################################################################
import pandas as pd # dataset load
import numpy as np
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import linkage, dendrogram
import math
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram

data=([1.6,55.5,1.8,62.8,4.6,67.7,2.7,68.8,3.8,61.3,6.4,56.4,5.2,61.1,1.2,63.4,-1.2,79.3],\
    [2.5,58,2.5,64.6,5.3,66.7,3.6,68,4.9,58.6,7.1,55.1,6,59.1,2.5,60,0,78.7],\
    [7.7,46.3,7.1,56.4,9,57.9,8.5,56.2,9.5,60.6,10.4,56.9,9.6,58.5,7.8,49.4,5.5,62.6],\
    [11.1,50,10.4,56.6,11.5,52.2,11.5,51.2,12.6,53.4,12.6,52.6,12.3,54.6,11,43.5,9.1,57.7],\
    [18,67.3,16.7,71.3,18.8,73.4,18.8,70.7,19.4,64,17.9,72.5,18.1,71.6,18.2,59.9,16.9,79.1],\
    [23.9,67.4,21.8,77,23.6,79.4,24.1,75.9,24.5,63.2,22.4,73.4,22.7,75,24.1,58.9,22.5,79.3],\
    [24.1,76.3,23.2,79.7,23.4,91.1,23.6,87.6,23.2,78.8,22.1,87.3,22.1,90.8,23.7,69.7,22.8,88.9],\
    [26.5,85.1,25.9,83.9,27.6,89,27.5,86.5,28.6,71.1,27.2,82.4,27.6,83.3,26.1,79.4,25.1,96],\
    [21.4,70.8,21.3,69.4,21.3,84.5,21.2,80.4,21.2,73.3,21.9,74,21.1,81,19.9,72.9,18.8,87.5],\
    [14.3,60.4,14.8,58.2,15.7,64.1,14.2,70.4,15.4,63.5,17.3,57.8,15.9,65.6,12.7,66.3,10.9,78.1],\
    [8,63.7,8.3,61.7,10.5,63.1,8.6,68.7,9.7,58.9,12.4,51.8,11,59.5,7.3,61.4,5,77.6],\
    [-0.4,57.8,0.2,55.5,2.7,61.4,0.5,64.5,1.6,53.7,4.3,42.9,3.4,46.5,-1.4,59.1,-3.9,72.8])
df=pd.DataFrame(data,columns=[['서울','서울','인천','인천','광주','광주','대전','대전','대구','대구','부산','부산','울산','울산','원주','원주','철원','철원'],\
        ['평균기온','상대습도','평균기온','상대습도','평균기온','상대습도','평균기온','상대습도','평균기온','상대습도',\
        '평균기온','상대습도','평균기온','상대습도','평균기온','상대습도','평균기온','상대습도']], \
        index=['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'])

##################################################################################################################

#%%
# 1. 서울지역의 평균기온(x)과 상대습도(y)의 선형회귀식을 구하고 결정계수 및 상관계수를 구하라. 그리고 상관계수가 유의미한 값을 갖는지 유의성을 판단하라.

x=df['서울','평균기온'].values    # .values 로 np array 형식으로 추출
y=df['서울','상대습도'].values

'''
표본의 회귀식은  y = b0 +b1x    ;b0 = 표본회귀식의 절편계수, 모집단회귀모형 절편계수의 추정량
                               ;b1 = 표본회귀식의 기울기, 모집단회귀모형 기울기의 추정량

회귀식을 구하는것은 곧 b0,b1을 구하는 것이고,
최소제곱법(추정오차를 최소화 하는법)은 b0과 b1이 최소가 되는 값을 구하는 것과 같다.
b0과 b1이 최소제곱추정량이 되게끔 구하면 다음과 같이 된다(과정생략)

b1 = r*(sy/sx)                ; r: 상관계수  
                              ; sy : y의 표준편차
                              ; sx : x의 표준편차
b0 = ym - b1*xm               ; ym : y 평균 
                              ; xm : x 평균
위에서 상관계수 r 은 다음과 같이 구한다.
r = cov(x,y) / std(x)*std(y)  
결정계수(R2)는 x,y의 상관계수 r 을 제곱한 값과 같으므로 r^2 으로 구할수 있다.
'''
r = np.cov(x,y)[0, 1] / (np.std(x,ddof=1)*np.std(y,ddof=1))   # ddof = 1 : 표본표준편차
R2 = r*r # 결정계수
b1 = r*(np.std(x,ddof=1)/np.std(y,ddof=1)) 
b0 = np.mean(y) - (b1 * np.mean(x))        


'''
위의 계산 결과에 따라 표본의 회귀식은 다음과 같다.
y = 54.3 + 0.67x 

결정계수 = 0.60
상관계수 = 0.78
'''

'''
서울의 평균기온과 상대습도의 상관계수가 유의미한 값을 갖는지에 대한 검정은 다음과 같이 할 수 있다.
두 변수가 정규분포를 따른다는 가정하에 귀무가설이 사실일때(r = 0, 상관관계없다.) 자유도가 n-2 인 t 분포를 따른다. 
T = r / sqrt((1-r*r)/n-2) , 이때 T 가 [-t(a/2),+t(a/2)] 사이에 있으면 귀무가설을 채택한다. (a=유의수준)
'''

n= len(x)
T = r / math.sqrt( (1-r*r) / (n-2) )

'''
위에서 구한 T = 3.91
t 분포표에 따르면
자유도 10 일때, 유의 수준 99% 일때 t 값이 2.764 이고 T는 약 3.9 이므로
귀무가설을 기각하여 서울의 평균기온과 상대습도의 상관계수는 유의미한 상관관계가 있다고 판단할 수 있다.
'''


del x,y,r,R2,b1,b0,n,T  # 변수 지우기
#%%
# 2. 원주와 철원의 상대습도가 서로 다르다고 할 수 있는지 t-test와 Mann-Whitney U-test 를 이용하여 유의성 5%에서 검정하라

# 2번 풀이, T-test
n=12
x=df['원주','상대습도'].values    
y=df['철원','상대습도'].values


'''
각 도시의 상대습도는 관측소의 관측치 이므로 표본값이라고 할 수 있다.
때문에 모분산을 알 수 없어 표본추정량을 사용하게되고 그 통계량은 자유도 n-1 인 t-분포를 따른다.
통계량은 다음과 같이 구할 수 있다.
T = (Xm - M) / (s/sqrt(n))  ; Xm = 표본 x의 평균 
                            ; M  = 모평균
                            ; s  = 표본표준편차
                            ; n  = 표본수
이번 경우에는 원주와 철원의 상대습도가 다른지 확인하는 것 이므로 모집단을 철원으로 가정하여 구한다.
그러면 모집단(철원)과 원주의 상대습도가 유의미한 차이가 있는지 검정 할 수 있다.
'''

T = (np.mean(x) - np.mean(y)) / (   np.std(x,ddof=1) / math.sqrt(n)   )

'''
위의 식으로 계산하면 T = -5.78 이고 t분포표(자유도=11, 양측검정)에 따르면 유의수준 5%에서의 통계량은 2.201
T의 절대값이 2.201보다 크므로 유의수준 5%에서 유의미한 차이가 있다고 할 수 있다.
'''

# 2번 풀이, Mann-Whitney U-test
X = np.zeros((12,2))
X[:,1] = x
X[:,0] = 0          # 원주 인덱스 = 0

Y = np.zeros((12,2))
Y[:,1] = y
Y[:,0] = 1          # 철원 인덱스 = 1

U = [X]+[Y]
U=np.array(U).reshape(24,2)

idx=U[:,1].argsort()
U=U[idx][::-1]

A=np.zeros((24,3))
for i in range(1,25):
    A[i-1]=np.append(U[i-1],i)


 #   R_chul1 = 철원의 순위합
 #   R_1ju   = 원주의 순위합
R_chul1=0 
R_1ju=0    
for i in range(24):
    if A[i][0] == 1:
       R_chul1 +=  A[i][2]  
    if A[i][0] == 0:
       R_1ju +=  A[i][2]           

'''
U1 = (n1 * n2) + (n1*(n1+1))/2 - R1     ; R1 : 1번표본의 순위합        
U2 = (n1 * n2) + (n2*(n2+1))/2 - R2     ; R2 : 2번표본의 순위합
U = max(U1,U2)
m = (U1+U2)/2 = (n1*n2)/2
s = sqrt( (n1*n2*(n1+n2+1)) /12 )

위와 같을때 확률변수 Z는 다음과 같은 정규분포를 따른다.
Z = (U-m) / s  ~ N(0,1)
'''

n1=12
n2=12
U_chul1 = (n1 * n2) + (n1*(n1+1))/2 - R_chul1
U_1ju = (n1 * n2) + (n2*(n2+1))/2 - R_1ju
U = max(U_chul1,U_1ju)
m = (U_chul1+U_1ju)/2 
s = math.sqrt( (n1*n2*(n1+n2+1)) / 12 )

Z = (U-m) / s

'''
위의 계산에 따르면 Z = 2.83 이다. 표준정규분포표에서 P(Z<=2.83) = 0.9977 이다.
따라서 유의미한 차이가 있다고 할 수 있다.
'''

del X,Y,x,y,n1,n2,U_1ju,U_chul1,U,m,s,Z,A,a,i,j  # 변수 지우기

#%%
# 3. 위 지역별 평균 기온을 EOF 분석하여 3번째 모드 까지의 평균기온의 공간분포와 시계열을 나타내어라
seoul=df['서울','평균기온'].values    
incheon=df['인천','평균기온'].values
gwangju=df['광주','평균기온'].values
daejeon=df['대전','평균기온'].values
daegu=df['대구','평균기온'].values
busan=df['부산','평균기온'].values
ulsan=df['울산','평균기온'].values
oneju=df['원주','평균기온'].values
chulone=df['철원','평균기온'].values


Temp =[seoul] + [incheon] + [gwangju] + [daejeon] + [daegu] + [busan] + [ulsan] + [oneju] + [chulone]
T=np.array(Temp)
T=np.transpose(T)


######################################################################
T = T - np.mean(T,axis=0)
cov_matrix= (T.T @ T) / 12

eig_vals, eig_vecs = np.linalg.eig(cov_matrix)
idx = eig_vals.argsort()[::-1] # 내림차순정렬 , D.argsort() = 오름차순정렬
eig_vals = eig_vals[idx]
eig_vecs = eig_vecs[:,idx]

###########################################################
# T_std = StandardScaler().fit_transform(T)
# covariance_matrix = np.cov(T_std)

# eig_vals, eig_vecs = np.linalg.eig(covariance_matrix)
# idx = eig_vals.argsort()[::-1] # 내림차순정렬 , D.argsort() = 오름차순정렬
# eig_vals = eig_vals[idx]
# eig_vecs = eig_vecs[:,idx]
########################################################################

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# eig_vals[0] / sum(eig_vals)

pc  = T @ eig_vecs

eof1=np.reshape(pc[:,0],(-1,1)) @ np.reshape(eig_vecs[:,0],(1,-1))
eof2=np.reshape(pc[:,1],(-1,1)) @ np.reshape(eig_vecs[:,1],(1,-1))
eof3=np.reshape(pc[:,2],(-1,1)) @ np.reshape(eig_vecs[:,2],(1,-1))



plt.figure(figsize=[12,9])

ax1=plt.subplot(3,2,1)
plt1=plt.plot(eof1[0,:],label='eof1')
ax1.set_title('EOF1')

ax2=plt.subplot(3,2,3)
plt2=plt.plot(eof2[0,:],label='eof2')
ax2.set_title('EOF2')

ax3=plt.subplot(3,2,5)
plt3=plt.plot(eof3[0,:],label='eof3')
ax3.set_title('EOF3')


ax11=plt.subplot(3,2,2)
plt11=plt.plot(pc[:,0])
ax11.set_title('PC1 Timeseires')

ax22=plt.subplot(3,2,4)
plt22=plt.plot(pc[:,1])
ax22.set_title('PC2 Timeseires')

ax33=plt.subplot(3,2,6)
plt33=plt.plot(pc[:,2])
ax33.set_title('PC3 Timeseires')

plt.tight_layout()



del T,cov_matrix,D,V,idx,pc,eof1,eof2,eof3,seoul,incheon,gwangju,daegu,daejeon,busan,ulsan,oneju,chulone

#%%

# 4. 1월달 평균기온과 상대습도를 기준으로 군집분석을 하고자 한다. 전체 지역의 1월 평균기온과 상대습도를 각각 표준화하고,  유사성은 (x,y)=(1월 표준화된 평균기온, 1월 표준화된 상대습도) 좌표계의 유클리드 거리로 판단하고 (거리가 작으면 유사성 높음), 평균연결법을 이용하여 9지역에 대한 계층적 군집 분포에 대한 덴드로그램을 작성하여라.
df2=df.stack(level=0)
T=df2['평균기온'][0:9].values # 광주 대구 대전 부산 서울 울산 원주 인천 철원
H=df2['상대습도'][0:9].values #

# 표준화
Tm=np.mean(T)
Tstd=np.std(T,ddof=1)
t = (T - Tm) / Tstd

Hm=np.mean(H)
Hstd=np.std(H,ddof=1)
h = (H - Hm) / Hstd

# 클러스터링 
X = np.array( [[ t[0] , h[0] ] , [ t[1] , h[1] ] , [ t[2] , h[2] ] , [ t[3] , h[3] ] , [ t[4] , h[4] ] ,\
    [ t[5] , h[5] ] , [ t[6] , h[6] ] , [ t[7] , h[7] ] , [ t[8] , h[8] ]] )
clusters = linkage(X, method='average', metric='euclidean')
clusters
clusters.shape 
plt.figure()
plt.title('dendrogram')
dendrogram(clusters, labels =['Gwangju', 'Daegu' ,'Daejeon', 'Busan ','Seoul', 'Ulsan', 'Oneju', 'Incheon','Chulone'],leaf_rotation=90, leaf_font_size=12)

#%%
# 5.지역별 1년간 상대습도와 기온간의 상관계수를 유사성(상관계수가 높으면 유사성 높음)으로 하여 평균연결법을 이용하여 9개 지역에 대한 계측적 군집 분포에 대한 덴드로그램을 작성하라

# Temp dendrogram
seoul=df['서울','평균기온'].values    
incheon=df['인천','평균기온'].values
gwangju=df['광주','평균기온'].values
daejeon=df['대전','평균기온'].values
daegu=df['대구','평균기온'].values
busan=df['부산','평균기온'].values
ulsan=df['울산','평균기온'].values
oneju=df['원주','평균기온'].values
chulone=df['철원','평균기온'].values

Temp =[seoul] + [incheon] + [gwangju] + [daejeon] + [daegu] + [busan] + [ulsan] + [oneju] + [chulone]
T=np.array(Temp)
T=np.transpose(T)

df3 = pd.DataFrame(T,columns=['서울','인천','광주','대전','대구','부산','울산','원주','철원'],index=['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'])

clusters = linkage(df3.T, method='average',metric='correlation')
plt.figure()
plt.title('Temp dendrogram')
dendrogram(clusters,labels=['Seoul','Incheon','Gwangju','Daejeon','Daegu','Busan','Ulsan','Oneju','Chulone'],leaf_rotation=90, leaf_font_size=12)

# RH dendrogram
seoul=df['서울','상대습도'].values    
incheon=df['인천','상대습도'].values
gwangju=df['광주','상대습도'].values
daejeon=df['대전','상대습도'].values
daegu=df['대구','상대습도'].values
busan=df['부산','상대습도'].values
ulsan=df['울산','상대습도'].values
oneju=df['원주','상대습도'].values
chulone=df['철원','상대습도'].values

Hum =[seoul] + [incheon] + [gwangju] + [daejeon] + [daegu] + [busan] + [ulsan] + [oneju] + [chulone]
H=np.array(Hum)
H=np.transpose(H)

df3 = pd.DataFrame(H,columns=['서울','인천','광주','대전','대구','부산','울산','원주','철원'],index=['1월','2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'])
# corr = df3.corr().values

clusters = linkage(df3.T, method='average',metric='correlation')
plt.figure()
plt.title('RH dendrogram')
dendrogram(clusters,labels=['Seoul','Incheon','Gwangju','Daejeon','Daegu','Busan','Ulsan','Oneju','Chulone'],leaf_rotation=90, leaf_font_size=12)
