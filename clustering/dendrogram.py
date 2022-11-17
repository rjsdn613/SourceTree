"""
계층적 군집분석
 - 유클리드 거리계산식 이용
 - 상향식(Bottom-up)으로 군집을 형성
"""

import pandas as pd # dataset load
from sklearn.datasets import load_iris
# 계층적 군집 model
from scipy.cluster.hierarchy import linkage, dendrogram



# 1. dataset load

iris = pd.read_csv("E:/CSL/visual studio/SourceTree/clustering/iris.csv")
iris.info()

cols = list(iris.columns)
iris_x = iris[cols[:4]]
iris_x.head()

 
iris['Species'].value_counts() # 'Species' : y변수
'''
versicolor    50
virginica     50
setosa        50
'''


# 2. y변수 수치화
X, y = load_iris(return_X_y=True)

# 사이킷런 라이브러리에서 제공하는 데이터셋을 불러오면 범주값을 숫자로 받을 수 있음
y  # 0,1,2로 구성됨

labels = pd.DataFrame(y, columns = ['labels'])

# df = df + df
irisDF = pd.concat([iris_x, labels], axis = 1)
irisDF.head()
irisDF.tail()  # x변수들과 수치화된 y변수(labels)로 데이터프레임 만들어진 것을 확인



# 3. 계층적 군집분석 model

clusters = linkage(y=irisDF, method='complete', metric='euclidean')
clusters
clusters.shape # (149, 4)

'''
연결방식 
 1. 단순연결방식(single)   : 두 클러스터상에서 가장 가까운 거리를 측정
 2. 완전연결방식(complete) : 두 클러스터상에서 가장 먼 거리를 측정
 3. 평균연결방식(average)  : 각 클러스터내의 각 점에서 다른 클러스터내의 모든 점사이의 거리에 대한 평균을 측정
'''


# 4. 덴드로그램 시각화 : 군집수 결정

import matplotlib.pyplot as plt
plt.figure( figsize = (25, 10) )
dendrogram(clusters, leaf_rotation=90, leaf_font_size=12,)
# leaf_rotation=90 : 글자 각도
# leaf_font_size=20 : 글자 사이즈
# plt.show() 
plt.savefig("E:/CSL/visual studio/SourceTree/clustering/dendrogram.png")
plt.close()

# 5. 클러스터링(군집) 결과
from scipy.cluster.hierarchy import fcluster # 지정한 클러스터 자르기

cut_tree = fcluster(clusters, t=3, criterion='distance')
cut_tree # prediction

labels = irisDF['labels'] # 정답

df = pd.DataFrame({'pred':cut_tree, 'labels':labels})

con_mat = pd.crosstab(df['pred'], df['labels'])
con_mat
'''
labels   0   1   2
pred              
1       50   0   0
2        0   0  34
3        0  50  16
'''

# irisDF에 군집 예측치 추가
irisDF.head()
irisDF['cluster'] = cut_tree
irisDF.head()

for a in range(150):
    if irisDF['cluster'][a] == 1:
        A=plt.scatter(x=irisDF['Sepal.Length'][a], y=irisDF['Petal.Length'][a], c='red')


for a in range(150):
    if irisDF['cluster'][a] == 2:
        B=plt.scatter(x=irisDF['Sepal.Length'][a], y=irisDF['Petal.Length'][a], c='blue')

        
for a in range(150):
    if irisDF['cluster'][a] == 3:
        C=plt.scatter(x=irisDF['Sepal.Length'][a], y=irisDF['Petal.Length'][a], c='purple')
        plt.legend([A,B,C],["cluster1(50)","cluster2(34)","cluster3(66)"])

plt.savefig("E:/CSL/visual studio/SourceTree/clustering/scatter.png")
plt.close()
# 클러스터 빈도수
irisDF['cluster'].value_counts()
'''
3    66
1    50
2    34
'''

# 각 클러스터별 통계(평균)
cluster_g = irisDF.groupby('cluster')
cluster_g.mean()
'''
         Sepal.Length  Sepal.Width  Petal.Length  Petal.Width    labels
cluster                                                                
1            5.006000     3.428000      1.462000     0.246000  0.000000
2            6.888235     3.100000      5.805882     2.123529  2.000000
3            5.939394     2.754545      4.442424     1.445455  1.242424
'''