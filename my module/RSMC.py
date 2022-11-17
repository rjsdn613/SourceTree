'''
많이 쓰는 기능들
- 특정 년도 태풍들 or 특정 태풍들의 특정위도에서의 강도 list
- 특정 년도 태풍들 or 특정 태풍들의 LMI 강도 list
- 특정 년도 태풍들 or 특정 태풍들의 dataframe

- 특정 년도 태풍들 or 특정 태풍들의 평균 트랙(시작/LMI/소멸 위치표시)


- 특정 년도 태풍들의 (발생시각{date} / 중위도도달시각{date} / 소멸시각{date}) --> date로 배경장 그리기
'''
#Library
import netCDF4 as nc
import pandas as pd
import numpy as np
from numpy import linspace
from collections import Counter
from random import *
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from scipy import interpolate

print("t=0 이면 년도로 호출, t=1 이면 태풍번호로 호출")
print("모든 년도와 태풍번호는 과거부터 현재순으로 입력해야 합니다.")
print("input(*args)이 list이면 앞에 *를 붙이세요.")
print("ex) tc_number=[1210, 1219, 1313] 일 때 사용법 => RSMC_I(*tc_number, t=1)")
print(" ")



if __name__ == "__main__":
    print("***** 사용법 *****")
    print("t=0 이면 년도로 호출, t=1 이면 태풍번호로 호출")
    print("모든 년도와 태풍번호는 과거부터 현재순으로 입력해야 합니다.")
    print("input(*args)이 list이면 앞에 *를 붙이세요.")
    print("ex) tc_number=[1210, 1219, 1313] 일 때 사용법 => RSMC_I(*tcs, t=1)")
    print(" ")
    print("----- Class 1 -----")
    print("RSMC_I : 태풍강도 출력 (하위 def : LMI, LAT) ")
    print("LMI : 태풍의 LMI 출력")
    print("LAT : 입력위도값에서 태풍의 최대풍속 출력")
    print(" ")
    print("ex) model = RSMC_I(2000,2001,t=0)")
    print("    model.LMI()  --> 2000년~2001년 태풍의 평균 LMI 출력")
    print("    model.LAT(30) --> 2000년~2001년 태풍의 북위30도에서 평균최대풍속 출력")
    print(" ")
    print("** 년도를 2000, 2005로 입력하면 2000~2005년")
    print("** 년도를 2000, 2003, 2012로 입력하면 2000, 2003, 2012년")
    print(" ")
    print("ex) model = RSMC_I(1218,1219,1512,t=1)")
    print("    model.LMI()  --> 태풍번호 1218, 1219,1512의 LMI 각각 출력")
    print("    model.LAT(30) --> 태풍번호 1218, 1219,1512의 북위30도에서 최대풍속 각각 출력")
    print("-------------------")
    print(" ")


    print("----- Class 2 -----")
    print("RSMC_T : 태풍트랙 출력 (하위 def : plot_track, plot_mean_track, plot_ts_track, plot_mean_ts_track)")
    print(" ")
    print("plot_track : 년도와 태풍번호로 각각의 태풍 트랙 출력 (LMI 위치 표시)")
    print("plot_mean_track : 년도와 태풍번호로 평균 트랙 출력 (LMI 위치 표시)")
    print("plot_ts_track : 년도와 태풍번호로 각각의 트랙 출력 (TS급 이상만) (LMI 위치 표시)")
    print("plot_mean_ts_track : 년도와 태풍번호로 평균 트랙 출력 (TS급 이상만) (LMI 위치 표시)")
    print("-------------------")


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def interp_method(data):
    from scipy import interpolate
    import numpy as np
    for i in range(len(data.columns)):
        x = data[i].dropna().index.to_list()
        y = data[i].dropna().values
        f_linear = interpolate.interp1d(x, y, kind='linear')
        xnew = np.linspace(0, x[-1], num=data.index[-1]+1, endpoint=True)
        y_new_linear = f_linear(xnew)
        data[i] = y_new_linear


#1021개의 태풍
def RSMC_tc_df(idx):
    if idx > 1021:
        print("!! 1021번이 마지막 태풍 !!")
    else: 
        print("!!! TS급 이상만 출력(3시간 간격 자료 간헐적존재) !!!")
    with open("E:/CSL/bst_all_82.txt", "r") as f:
        TCs=0
        while True:

            line = f.readline()
            if not line:
                break
            TCs += 1 # 전체 태풍중 몇번째 태풍인지 (연도 상관없이)
            TC_info_line = line.split()
            # TC_number = int(TC_info_line[1][2:5])
            TC_number = int(TC_info_line[1])
            TC_count_num = int(TC_info_line[2])
            if idx == TCs:
                if int(TC_info_line[1][0:2]) >= 50:
                    yrs_idx = 1900
                elif int(TC_info_line[1][0:2]) <= 50:
                    yrs_idx = 2000
                # print("TC number is", TC_number, "in", int(TC_info_line[1][0:2])+yrs_idx )
                # print("grade 2 : Tropical Depression (TD)   \
                #        grade 3 : Tropical Storm (TS)        \
                #        grade 4 : Severe Tropical Storm (STS)\
                #        grade 5 : Typhoon (TY)               \
                #        grade 6 : Extra-tropical Cyclone (L) \
                #        grade 9 : Tropical Cyclone of TS intensity or higher ")
                TC_data = []
                for i in range(TC_count_num):
                    data = f.readline()
                    data = data.split()
                    data[6] = round(int(data[6])*0.514,2) # knots to m/s, 소수점 2자리까지 표현
                    data[3] = round(int(data[3])*0.1,2) # lat
                    data[4] = round(int(data[4])*0.1,2) # lon
                    TC_data.append(data) 
                    tc=pd.DataFrame(TC_data)   
                    if len(tc.columns) == 7:
                        tc.columns=['date','indicator 002','grade','lat','lon','pres','spd']    
                        tc=tc[['date','grade','lat','lon','pres','spd']]   
                        tc.insert(0, 'num', TC_number)      
                        tc.insert(0, 'idx', idx)      
                                
                    elif len(tc.columns) == 11:
                        tc.columns=['date','indicator 002','grade','lat','lon','pres','spd','1','2','3','4']    
                        tc=tc[['date','grade','lat','lon','pres','spd']]   
                        tc.insert(0, 'num', TC_number)      
                        tc.insert(0, 'idx', idx)      

                    elif len(tc.columns) == 12:
                        tc.columns=['date','indicator 002','grade','lat','lon','pres','spd','1','2','3','4','5']    
                        tc=tc[['date','grade','lat','lon','pres','spd']]   
                        tc.insert(0, 'num', TC_number)      
                        tc.insert(0, 'idx', idx)      
                                    
            else: 
                for i in range(TC_count_num): 
                    f.readline()
    return tc[tc.spd >= 17]




#Intensity
#애초에 TS급 이상의 데이터만 넣는게 좋은가?
class RSMC_I:
    def __init__(self, *args, t): # t=0이면 년도로 호출, t=1이면 태풍번호로 호출

        self.args = args
        self.t = t
        self.idxs = []

        #년도로 호출
        if t == 0:
            if len(args) < 3:
                self.sidx = args[0]
                self.eidx = args[-1]
                yrs = self.sidx
                for i in range(self.eidx-self.sidx+1):
                    self.idxs.append(yrs)             
                    yrs += 1

            elif len(args) >= 3:
                for i in range(len(args)):
                    self.idxs.append(args[i]) 

            return print("선택 년도: ",self.idxs)

        #태풍번호로 호출 (과거부터 현재순으로 호출해야함)
        elif t == 1:
            for i in range(len(args)):
                self.idxs.append(args[i]) 

            return print("선택 태풍번호: ", self.idxs)

    def __call__(self):
        return print("선택 년도: ", self.idxs)


    def LMI(self): # t=0이면 년도로 호출, t=1이면 태풍번호로 호출
        idx = self.idxs
        pLMI = []
        LMI = []
        
        with open("E:/CSL/bst_all_82.txt", "r") as f:

                for i in range(len(idx)):

                    if i == 0 :
                        line = f.readline()
                        TC_info_line = line.split()
                        TC_number = int(TC_info_line[1])
                        TC_count_num = int(TC_info_line[2])

                    if self.t == 0 :
                        yrs = idx[i]
                        while TC_number != ((yrs % 100) * 100 + 1):
                            for a in range(TC_count_num):
                                line = f.readline()

                            line = f.readline()
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])

                    elif self.t == 1 :
                        tc_nums = idx[i]
                        while TC_number != tc_nums:
                            for a in range(TC_count_num):
                                line = f.readline()
                            line = f.readline()
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])


                    
                    if self.t == 0 :                    
                        yr_idx = int(TC_number / 100)
                        if yr_idx < 30:
                            yr_idx += 2000
                        else:
                            yr_idx += 1900

                        while True and yrs == yr_idx:
                            wspd = []
                            import numpy as np
                            for i in range(TC_count_num):
                                data = f.readline()
                                spd = round(int(data.split()[6])*0.514, 2) #wspd(m/s)
                                wspd.append(spd)
                            pLMI.append(np.max(wspd))

                            line = f.readline()
                            if not line:
                                break
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])

                            yr_idx = int(TC_number / 100)
                            if yr_idx < 30:
                                yr_idx += 2000
                            else:
                                yr_idx += 1900
                        LMI.append(np.mean(pLMI))


                    elif self.t == 1 :
                        while True and tc_nums == TC_number:
                            wspd = []
                            import numpy as np
                            for i in range(TC_count_num):
                                data = f.readline()
                                spd = round(int(data.split()[6])*0.514, 2) #wspd(m/s)
                                wspd.append(spd)
                            pLMI = np.max(wspd)

                            line = f.readline()
                            if not line:
                                break
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])

                        LMI.append(pLMI)
        return (idx, LMI)


    def LAT(self, lat):
        idx = self.idxs
        self.lat = lat
        mean_lat_spd = []

        with open("E:/CSL/bst_all_82.txt", "r") as f:

                for i in range(len(idx)):

                    if i == 0 :
                        line = f.readline()
                        TC_info_line = line.split()
                        TC_number = int(TC_info_line[1])
                        TC_count_num = int(TC_info_line[2])

                    if self.t == 0 :
                        yrs = idx[i]
                        while TC_number != ((yrs % 100) * 100 + 1):
                            for a in range(TC_count_num):
                                line = f.readline()

                            line = f.readline()
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])

                    elif self.t == 1 :
                        tc_nums = idx[i]
                        while TC_number != tc_nums:
                            for a in range(TC_count_num):
                                line = f.readline()
                            line = f.readline()
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])


                    
                    if self.t == 0 :                    
                        yr_idx = int(TC_number / 100)
                        if yr_idx < 30:
                            yr_idx += 2000
                        else:
                            yr_idx += 1900

                        while True and yrs == yr_idx:
                            wspd = []
                            lats = []
                            lat_spd = []
                            for i in range(TC_count_num):
                                data = f.readline()
                                spd = round(int(data.split()[6])*0.514, 2) #wspd(m/s)
                                tclat = float(data.split()[3])*0.1 #lat
                                wspd.append(spd)
                                lats.append(tclat)
                            lat_idx = lats.index(find_nearest(lats, self.lat))
                            if lats[lat_idx] - self.lat > 3:
                                print("입력 위도값과 가장 가까운 태풍의 위도값이 3도이상 차이나므로 주의하세요.")
                            lat_spd.append(wspd[lat_idx])

                            line = f.readline()
                            if not line:
                                break
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])

                            yr_idx = int(TC_number / 100)
                            if yr_idx < 30:
                                yr_idx += 2000
                            else:
                                yr_idx += 1900
                        mean_lat_spd.append(round(np.mean(lat_spd), 2))


                    elif self.t == 1 :
                        while True and tc_nums == TC_number:
                            print("베스트트랙 태풍번호", TC_number)
                            print("입력 태풍번호", tc_nums)
                            wspd = []
                            lats = []
                            lat_spd = []
                            counts=0
                            for i in range(TC_count_num):
                                data = f.readline()
                                spd = round(int(data.split()[6])*0.514, 2) #wspd(m/s)
                                tclat = float(data.split()[3])*0.1 #lat
                                wspd.append(spd)
                                lats.append(tclat)

                            if len(np.where(np.array(lats)==find_nearest(lats, self.lat))[0]) == 1:
                                lat_idx = np.where(np.array(lats)==find_nearest(lats, self.lat))[0][0]
                            elif len(np.where(np.array(lats)==find_nearest(lats, self.lat))[0]) > 1:
                                cts = len(np.where(np.array(lats)==find_nearest(lats, self.lat))[0])
                                x = int(input("태풍이 입력위도값을 "+str(cts)+"번 지나갑니다. 몇 번째 위도값을 반환할지 입력하세요(0부터 시작) : "))
                                lat_idx = np.where(np.array(lats)==find_nearest(lats, self.lat))[0][x]


                            if lats[lat_idx] - self.lat > 3:
                                counts+=1
                            lat_spd.append(wspd[lat_idx])

                            line = f.readline()
                            if not line:
                                break
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])
                        if counts >= 1:
                            print("입력 위도값과 가장 가까운 태풍[",str(counts),"] 의 위도값이 3도이상 차이나므로 주의하세요.")
                        mean_lat_spd.append(round(np.mean(lat_spd), 2))

                print("입력위도값에서 태풍의 풍속 값 (0인 경우, missing data)")
        return (idx, mean_lat_spd)




#Track
#이것도 애초에 그냥 TS급 이상의 데이터를 넣을까?
class RSMC_T:
    def __init__(self, *args, t): # t=0이면 년도로 호출, t=1이면 태풍번호로 호출
        print("t=0 이면 년도로 호출, t=1 이면 태풍번호로 호출")
        print("모든 년도와 태풍번호는 과거부터 현재순으로 입력해야 합니다.")
        print("input(*args)이 list이면 앞에 *를 붙이세요.")
        print("ex) tc_number=[1210, 1219, 1313] 일 때 사용법 => RSMC_T(*tcs, t=1) \n")

        print("--------------------- def 종류 --------------------------")
        print("plot_track : 년도와 태풍번호로 각각의 태풍 트랙 출력 (LMI 위치 표시)")
        print("plot_mean_track : 년도와 태풍번호로 평균 트랙 출력 (LMI 위치 표시)")
        print("plot_ts_track : 년도와 태풍번호로 평균 트랙 출력 (TS급 이상만) (LMI 위치 표시)")
        print("plot_mean_ts_track : 년도와 태풍번호로 평균 트랙 출력 (TS급 이상만) (LMI 위치 표시)")
        print("--------------------------------------------------------")
        
        self.args = args
        self.t = t
        self.idxs = []

        #년도로 호출
        if t == 0:
            if len(args) < 3:
                self.sidx = args[0]
                self.eidx = args[-1]
                yrs = self.sidx
                for i in range(self.eidx-self.sidx+1):
                    self.idxs.append(yrs)             
                    yrs += 1

            elif len(args) >= 3:
                for i in range(len(args)):
                    self.idxs.append(args[i]) 

            return print("선택 년도: ",self.idxs)

        #태풍번호로 호출 (과거부터 현재순으로 호출해야함)
        elif t == 1:
            for i in range(len(args)):
                self.idxs.append(args[i]) 

            return print("선택 태풍번호: ", self.idxs)

    def __call__(self):
        return print("선택 년도: ", self.idxs)


    def plot_track(self):
        idx = self.idxs

        #년도로 호출
        if self.t == 0 :

            with open("E:/CSL/bst_all_82.txt", "r") as f:

                    fig, ax = plt.subplots(1,1, figsize=(10,10))
                    map = Basemap(projection='mill',
                    llcrnrlat=0, 
                    urcrnrlat=50,
                    llcrnrlon=100, 
                    urcrnrlon=180 )

                    map.drawcoastlines()
                    map.drawmeridians(np.arange(0, 360, 30), labels=[0,0,0,1])
                    map.drawparallels(np.arange(-90, 90, 10), labels=[1,0,0,0])

                    lines = []
                    for i in range(len(idx)):

                        if i == 0 :
                            line = f.readline()
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])

                        yrs = idx[i]
                        while TC_number != ((yrs % 100) * 100 + 1):
                            for a in range(TC_count_num):
                                line = f.readline()
                            line = f.readline()
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])

                        yr_idx = int(TC_number / 100)

                        if yr_idx < 30:
                            yr_idx += 2000
                        else:
                            yr_idx += 1900
                        print("베스트트랙 년도: ", yr_idx)
                        print("입력 년도: ", yrs)

                        while True and yrs == yr_idx:
                            lats = []
                            lons = []
                            wspd = []
                            for i in range(TC_count_num):
                                data = f.readline()
                                tclat = float(data.split()[3])*0.1 #lat
                                tclon = float(data.split()[4])*0.1 #lon
                                spd = round(int(data.split()[6])*0.514, 2) #wspd(m/s)

                                wspd.append(spd)
                                lats.append(tclat)
                                lons.append(tclon)
                            #plot
                            tclon,tclat = map(lons,lats)
                            lines += plt.plot(tclon,tclat, '-o', markersize=4, label=str(TC_number))
                            LMI_loc = wspd.index(np.max(wspd))
                            tclmilon,tclmilat = map(lons[LMI_loc],lats[LMI_loc])
                            plt.plot(tclmilon,tclmilat, 'r^', markersize=10)

                            line = f.readline()
                            if not line:
                                break
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])

                            yr_idx = int(TC_number / 100)
                            if yr_idx < 30:
                                yr_idx += 2000
                            else:
                                yr_idx += 1900      
                    labels = [l.get_label() for l in lines]
                    plt.legend(lines, labels, loc='upper left')

        #태풍번호로 호출
        elif self.t == 1 :
            with open("E:/CSL/bst_all_82.txt", "r") as f:

                fig, ax = plt.subplots(1,1, figsize=(10,10))
                map = Basemap(projection='mill',
                llcrnrlat=0, 
                urcrnrlat=50,
                llcrnrlon=100, 
                urcrnrlon=180 )

                map.drawcoastlines()
                map.drawmeridians(np.arange(0, 360, 30), labels=[0,0,0,1])
                map.drawparallels(np.arange(-90, 90, 10), labels=[1,0,0,0])

                lines = []
                for i in range(len(idx)):

                    if i == 0 :
                        line = f.readline()
                        TC_info_line = line.split()
                        TC_number = int(TC_info_line[1])
                        TC_count_num = int(TC_info_line[2])

                    tc_nums = idx[i]
                    while TC_number != tc_nums:
                        for a in range(TC_count_num):
                            line = f.readline()
                        line = f.readline()
                        TC_info_line = line.split()
                        TC_number = int(TC_info_line[1])
                        TC_count_num = int(TC_info_line[2])

                    while True and tc_nums == TC_number:
                        print("베스트트랙 태풍번호", TC_number)
                        print("입력 태풍번호", tc_nums)
                        lats = []
                        lons = []
                        wspd = []
                        for i in range(TC_count_num):
                            data = f.readline()
                            tclat = float(data.split()[3])*0.1 #lat
                            tclon = float(data.split()[4])*0.1 #lon
                            spd = round(int(data.split()[6])*0.514, 2) #wspd(m/s)

                            lats.append(tclat)
                            lons.append(tclon)
                            wspd.append(spd)
                        #plot
                        tclon,tclat = map(lons,lats)
                        lines += plt.plot(tclon,tclat, '-o', markersize=4, label=str(TC_number))
                        LMI_loc = wspd.index(np.max(wspd))
                        tclmilon,tclmilat = map(lons[LMI_loc],lats[LMI_loc])
                        plt.plot(tclmilon,tclmilat, 'r^', markersize=10)    


                        line = f.readline()
                        if not line:
                            break
                        TC_info_line = line.split()
                        TC_number = int(TC_info_line[1])
                        TC_count_num = int(TC_info_line[2])

                labels = [l.get_label() for l in lines]
                plt.legend(lines, labels, loc='upper left')


    def plot_mean_track(self):

        idx = self.idxs

        #년도로 호출
        if self.t == 0 :

            with open("E:/CSL/bst_all_82.txt", "r") as f:

                    fig, ax = plt.subplots(1,1, figsize=(10,10))
                    map = Basemap(projection='mill',
                    llcrnrlat=0, 
                    urcrnrlat=50,
                    llcrnrlon=100, 
                    urcrnrlon=180 )

                    map.drawcoastlines()
                    map.drawmeridians(np.arange(0, 360, 30), labels=[0,0,0,1])
                    map.drawparallels(np.arange(-90, 90, 10), labels=[1,0,0,0])

                    lines = []
                    for i in range(len(idx)):

                        if i == 0 :
                            line = f.readline()
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])

                        yrs = idx[i]
                        while TC_number != ((yrs % 100) * 100 + 1):
                            for a in range(TC_count_num):
                                line = f.readline()
                            line = f.readline()
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])

                        yr_idx = int(TC_number / 100)

                        if yr_idx < 30:
                            yr_idx += 2000
                        else:
                            yr_idx += 1900
                        print("베스트트랙 년도: ", yr_idx)
                        print("입력 년도: ", yrs)


                        lats_df = pd.DataFrame([])
                        lons_df = pd.DataFrame([])
                        wspd_df = pd.DataFrame([])
                        while True and yrs == yr_idx:
                            lats=[]
                            lons=[]
                            wspd=[]
                            for i in range(TC_count_num):
                                data = f.readline()
                                tclat = float(data.split()[3])*0.1 #lat
                                tclon = float(data.split()[4])*0.1 #lon
                                spd = round(int(data.split()[6])*0.514, 2) #wspd(m/s)
                                wspd.append(spd)
                                lats.append(tclat)
                                lons.append(tclon)
                            wspd_df = pd.concat([wspd_df, pd.DataFrame(wspd)], join='outer', axis=1)
                            lats_df = pd.concat([lats_df, pd.DataFrame(lats)], join='outer', axis=1)
                            lons_df = pd.concat([lons_df, pd.DataFrame(lons)], join='outer', axis=1)

                            line = f.readline()
                            if not line:
                                break
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])
                            yr_idx = int(TC_number / 100)
                            if yr_idx < 30:
                                yr_idx += 2000
                            else:
                                yr_idx += 1900   

                        #columns index reset
                        wspd_df.columns = range(wspd_df.columns.size)
                        lats_df.columns = range(lats_df.columns.size)
                        lons_df.columns = range(lons_df.columns.size)

                        #interpolation
                        interp_method(wspd_df)
                        interp_method(lats_df)
                        interp_method(lons_df)
                        #mean
                        mean_wspd = np.mean(wspd_df, axis=1)
                        mean_lats = np.mean(lats_df, axis=1)
                        mean_lons = np.mean(lons_df, axis=1)

                        #plot
                        tclon,tclat = map(mean_lons,mean_lats)
                        lines += plt.plot(tclon,tclat, '-', markersize=4, label=str(yr_idx-1)) #위의 알고리즘 순서상 (년도+1)이되기 때문에 -1로 맞춰줌
                        LMI_loc = list(mean_wspd).index(np.max(mean_wspd))
                        tclmilon,tclmilat = map(mean_lons[LMI_loc],mean_lats[LMI_loc])
                        plt.plot(tclmilon,tclmilat, 'r^', markersize=10)

                        labels = [l.get_label() for l in lines]
                        plt.legend(lines, labels, loc='upper left')

        #태풍번호로 호출
        elif self.t == 1 :
            with open("E:/CSL/bst_all_82.txt", "r") as f:

                fig, ax = plt.subplots(1,1, figsize=(10,10))
                map = Basemap(projection='mill',
                llcrnrlat=0, 
                urcrnrlat=50,
                llcrnrlon=100, 
                urcrnrlon=180 )

                map.drawcoastlines()
                map.drawmeridians(np.arange(0, 360, 30), labels=[0,0,0,1])
                map.drawparallels(np.arange(-90, 90, 10), labels=[1,0,0,0])

                lines = []
                lats_df = pd.DataFrame([])
                lons_df = pd.DataFrame([])
                wspd_df = pd.DataFrame([])
                for i in range(len(idx)):

                    if i == 0 :
                        line = f.readline()
                        TC_info_line = line.split()
                        TC_number = int(TC_info_line[1])
                        TC_count_num = int(TC_info_line[2])

                    tc_nums = idx[i]
                    while TC_number != tc_nums:
                        for a in range(TC_count_num):
                            line = f.readline()
                        line = f.readline()
                        TC_info_line = line.split()
                        TC_number = int(TC_info_line[1])
                        TC_count_num = int(TC_info_line[2])


                    while True and tc_nums == TC_number:
                        print("베스트트랙 태풍번호", TC_number)
                        print("입력 태풍번호", tc_nums)
                        lats=[]
                        lons=[]
                        wspd=[]
                        for i in range(TC_count_num):
                            data = f.readline()
                            tclat = float(data.split()[3])*0.1 #lat
                            tclon = float(data.split()[4])*0.1 #lon
                            spd = round(int(data.split()[6])*0.514, 2) #wspd(m/s)
                            wspd.append(spd)
                            lats.append(tclat)
                            lons.append(tclon)
                        wspd_df = pd.concat([wspd_df, pd.DataFrame(wspd)], join='outer', axis=1)
                        lats_df = pd.concat([lats_df, pd.DataFrame(lats)], join='outer', axis=1)
                        lons_df = pd.concat([lons_df, pd.DataFrame(lons)], join='outer', axis=1)
                        line = f.readline()
                        if not line:
                            break
                        TC_info_line = line.split()
                        TC_number = int(TC_info_line[1])
                        TC_count_num = int(TC_info_line[2])

                #columns index reset
                wspd_df.columns = range(wspd_df.columns.size)
                lats_df.columns = range(lats_df.columns.size)
                lons_df.columns = range(lons_df.columns.size)

                #interpolation
                interp_method(wspd_df)
                interp_method(lats_df)
                interp_method(lons_df)
                #mean
                mean_wspd = np.mean(wspd_df, axis=1)
                mean_lats = np.mean(lats_df, axis=1)
                mean_lons = np.mean(lons_df, axis=1)

                #plot
                tclon,tclat = map(mean_lons,mean_lats)
                lines += plt.plot(tclon,tclat, '-', markersize=4, label="Mean track") 
                LMI_loc = list(mean_wspd).index(np.max(mean_wspd))
                tclmilon,tclmilat = map(mean_lons[LMI_loc],mean_lats[LMI_loc])
                plt.plot(tclmilon,tclmilat, 'r^', markersize=10)
            labels = [l.get_label() for l in lines]
            plt.legend(lines, labels, loc='upper left')






#Datetime
class RSMC_D:
    def __init__(self, *args, t): # t=0이면 년도로 호출, t=1이면 태풍번호로 호출

        self.args = args
        self.t = t
        self.idxs = []

        #년도로 호출
        if t == 0:
            if len(args) < 3:
                self.sidx = args[0]
                self.eidx = args[-1]
                yrs = self.sidx
                for i in range(self.eidx-self.sidx+1):
                    self.idxs.append(yrs)             
                    yrs += 1

            elif len(args) >= 3:
                for i in range(len(args)):
                    self.idxs.append(args[i]) 

            return print("선택 년도: ",self.idxs)

        #태풍번호로 호출 (과거부터 현재순으로 호출해야함)
        elif t == 1:
            for i in range(len(args)):
                self.idxs.append(args[i]) 

            return print("선택 태풍번호: ", self.idxs)

    def __call__(self):
        return print("선택 년도: ", self.idxs)

    def start2end(self):
        idx = self.idxs
        with open("E:/CSL/bst_all_82.txt", "r") as f:

            start_date = []
            end_date = []
            for i in range(len(idx)):
                if i == 0 :
                    line = f.readline()
                    TC_info_line = line.split()
                    TC_number = int(TC_info_line[1])
                    TC_count_num = int(TC_info_line[2])

                if self.t == 0 :
                    yrs = idx[i]
                    while TC_number != ((yrs % 100) * 100 + 1):
                        for a in range(TC_count_num):
                            line = f.readline()

                        line = f.readline()
                        TC_info_line = line.split()
                        TC_number = int(TC_info_line[1])
                        TC_count_num = int(TC_info_line[2])

                elif self.t == 1 :
                    tc_nums = idx[i]
                    while TC_number != tc_nums:
                        for a in range(TC_count_num):
                            line = f.readline()
                        line = f.readline()
                        TC_info_line = line.split()
                        TC_number = int(TC_info_line[1])
                        TC_count_num = int(TC_info_line[2])


                if self.t == 0 :                    
                    yr_idx = int(TC_number / 100)
                    if yr_idx < 30:
                        yr_idx += 2000
                    else:
                        yr_idx += 1900

                    while True and yrs == yr_idx:
                        for i in range(TC_count_num):
                            data = f.readline()
                            if i == 0 :
                                start_date.append(int(data.split()[0]))
                            elif i == TC_count_num-1:
                                end_date.append(int(data.split()[0]))


                        line = f.readline()
                        if not line:
                            break
                        TC_info_line = line.split()
                        TC_number = int(TC_info_line[1])
                        TC_count_num = int(TC_info_line[2])

                        yr_idx = int(TC_number / 100)
                        if yr_idx < 30:
                            yr_idx += 2000
                        else:
                            yr_idx += 1900


                elif self.t == 1 :

                    while True and tc_nums == TC_number:
                        print("베스트트랙 태풍번호", TC_number)
                        print("입력 태풍번호", tc_nums)

                        for i in range(TC_count_num):
                            data = f.readline()
                            if i == 0 :
                                start_date.append(int(data.split()[0]))
                            elif i == TC_count_num-1:
                                end_date.append(int(data.split()[0]))


                        line = f.readline()
                        if not line:
                            break
                        TC_info_line = line.split()
                        TC_number = int(TC_info_line[1])
                        TC_count_num = int(TC_info_line[2])

        return (idx, start_date, end_date)


    def LAT(self, lat):
        idx = self.idxs
        self.lat = lat

        with open("E:/CSL/bst_all_82.txt", "r") as f:
                lat_dates = []
                for i in range(len(idx)):

                    if i == 0 :
                        line = f.readline()
                        TC_info_line = line.split()
                        TC_number = int(TC_info_line[1])
                        TC_count_num = int(TC_info_line[2])

                    if self.t == 0 :
                        yrs = idx[i]
                        while TC_number != ((yrs % 100) * 100 + 1):
                            for a in range(TC_count_num):
                                line = f.readline()

                            line = f.readline()
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])

                    elif self.t == 1 :
                        tc_nums = idx[i]
                        while TC_number != tc_nums:
                            for a in range(TC_count_num):
                                line = f.readline()
                            line = f.readline()
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])


                    
                    if self.t == 0 :
                        yr_idx = int(TC_number / 100)
                        if yr_idx < 30:
                            yr_idx += 2000
                        else:
                            yr_idx += 1900

                        while True and yrs == yr_idx:
                            dates = []
                            lats = []
                            for i in range(TC_count_num):
                                data = f.readline()
                                tclat = float(data.split()[3])*0.1 #lat
                                tcdate = int(data.split()[0])
                                lats.append(tclat)
                                dates.append(tcdate)

                            lat_idx = lats.index(find_nearest(lats, self.lat))
                            if lats[lat_idx] - self.lat > 3:
                                print("입력 위도값과 가장 가까운 태풍의 위도값이 3도이상 차이나므로 주의하세요.")
                            lat_dates.append(dates[lat_idx])

                            line = f.readline()
                            if not line:
                                break
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])

                            yr_idx = int(TC_number / 100)
                            if yr_idx < 30:
                                yr_idx += 2000
                            else:
                                yr_idx += 1900


                    elif self.t == 1 :
                        while True and tc_nums == TC_number:
                            print("베스트트랙 태풍번호", TC_number)
                            print("입력 태풍번호", tc_nums)
                            dates = []
                            lats = []
                            counts=0
                            for i in range(TC_count_num):
                                data = f.readline()
                                tcdate = int(data.split()[0])
                                tclat = float(data.split()[3])*0.1 #lat
                                lats.append(tclat)
                                dates.append(tcdate)

                            if len(np.where(np.array(lats)==find_nearest(lats, self.lat))[0]) == 1:
                                lat_idx = np.where(np.array(lats)==find_nearest(lats, self.lat))[0][0]
                            elif len(np.where(np.array(lats)==find_nearest(lats, self.lat))[0]) > 1:
                                cts = len(np.where(np.array(lats)==find_nearest(lats, self.lat))[0])
                                x = int(input("태풍이 입력위도값을 "+str(cts)+"번 지나갑니다. 몇 번째 위도값을 반환할지 입력하세요(0부터 시작) : "))
                                lat_idx = np.where(np.array(lats)==find_nearest(lats, self.lat))[0][x]


                            if lats[lat_idx] - self.lat > 3:
                                counts+=1
                            lat_dates.append(dates[lat_idx])

                            line = f.readline()
                            if not line:
                                break
                            TC_info_line = line.split()
                            TC_number = int(TC_info_line[1])
                            TC_count_num = int(TC_info_line[2])
                        if counts >= 1:
                            print("입력 위도값과 가장 가까운 태풍[",str(counts),"] 의 위도값이 3도이상 차이나므로 주의하세요.")

        return (idx, lat_dates)


