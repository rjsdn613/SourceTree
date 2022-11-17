"""
아래는 연도별로 한번에 불러올수있는 코드

"""


def get_tc_lat_yr(years):

    with open("E:/CSL/new_bst_all_80.txt", "r") as f:

        LAT = []

        line = f.readline()
        TC_info_line = line.split()
        TC_number = int(TC_info_line[1])
        TC_count_num = int(TC_info_line[2])

        while TC_number != ((years % 100) * 100 + 1):
            for a in range(TC_count_num):
                line = f.readline()

            line = f.readline()
            TC_info_line = line.split()
            TC_number = int(TC_info_line[1])
            TC_count_num = int(TC_info_line[2])

        a = 0

        while True:

            a += 1

            if a > 1:
                line = f.readline()
                if not line:
                    break
                TC_info_line = line.split()
                TC_number = int(TC_info_line[1])
                TC_count_num = int(TC_info_line[2])

            idx = int(TC_number / 100)
            if idx < 30:
                idx += 2000
            else:
                idx += 1900

            if years == idx:
                # LAT.append("===============================")
                # LAT.append(["TC_count_num is",TC_count_num])
                # LAT.append("===============================")
                for i in range(TC_count_num):
                    data = f.readline()
                    lat = int(data.split()[3]) * 0.1
                    LAT.append(lat)

    return LAT


def get_tc_lon_yr(years):

    with open("E:/CSL/new_bst_all_80.txt", "r") as f:

        LON = []

        line = f.readline()
        TC_info_line = line.split()
        TC_number = int(TC_info_line[1])
        TC_count_num = int(TC_info_line[2])

        while TC_number != ((years % 100) * 100 + 1):
            for a in range(TC_count_num):
                line = f.readline()

            line = f.readline()
            TC_info_line = line.split()
            TC_number = int(TC_info_line[1])
            TC_count_num = int(TC_info_line[2])

        a = 0

        while True:

            a += 1

            if a > 1:
                line = f.readline()
                if not line:
                    break
                TC_info_line = line.split()
                TC_number = int(TC_info_line[1])
                TC_count_num = int(TC_info_line[2])

            idx = int(TC_number / 100)
            if idx < 30:
                idx += 2000
            else:
                idx += 1900

            if years == idx:
                # LON.append("===============================")
                # LON.append(["TC_count_num is",TC_count_num])
                # LON.append("===============================")
                for i in range(TC_count_num):
                    data = f.readline()
                    lon = int(data.split()[4]) * 0.1
                    LON.append(lon)

    return LON


def get_tc_pres_yr(years):

    with open("E:/CSL/new_bst_all_80.txt", "r") as f:

        PRES = []

        line = f.readline()
        TC_info_line = line.split()
        TC_number = int(TC_info_line[1])
        TC_count_num = int(TC_info_line[2])

        while TC_number != ((years % 100) * 100 + 1):
            for a in range(TC_count_num):
                line = f.readline()

            line = f.readline()
            TC_info_line = line.split()
            TC_number = int(TC_info_line[1])
            TC_count_num = int(TC_info_line[2])

        a = 0

        while True:

            a += 1

            if a > 1:
                line = f.readline()
                if not line:
                    break
                TC_info_line = line.split()
                TC_number = int(TC_info_line[1])
                TC_count_num = int(TC_info_line[2])

            idx = int(TC_number / 100)
            if idx < 30:
                idx += 2000
            else:
                idx += 1900

            if years == idx:
                # PRES.append("===============================")
                # PRES.append(["TC_count_num is",TC_count_num])
                # PRES.append("===============================")
                for i in range(TC_count_num):
                    data = f.readline()
                    pres = int(data.split()[5])
                    PRES.append(pres)

    return PRES


def get_tc_wind_yr(years):

    with open("E:/CSL/new_bst_all_80.txt", "r") as f:

        WIND = []

        line = f.readline()
        TC_info_line = line.split()
        TC_number = int(TC_info_line[1])
        TC_count_num = int(TC_info_line[2])

        while TC_number != ((years % 100) * 100 + 1):
            for a in range(TC_count_num):
                line = f.readline()

            line = f.readline()
            TC_info_line = line.split()
            TC_number = int(TC_info_line[1])
            TC_count_num = int(TC_info_line[2])

        a = 0

        while True:

            a += 1

            if a > 1:
                line = f.readline()
                if not line:
                    break
                TC_info_line = line.split()
                TC_number = int(TC_info_line[1])
                TC_count_num = int(TC_info_line[2])

            idx = int(TC_number / 100)
            if idx < 30:
                idx += 2000
            else:
                idx += 1900

            if years == idx:
                # WIND.append("===============================")
                # WIND.append(["TC_count_num is",TC_count_num])
                # WIND.append("===============================")
                for i in range(TC_count_num):
                    data = f.readline()
                    wind = int(data.split()[6])*0.514
                    WIND.append(wind)

    return WIND


def get_tc_grade_yr(years):

    with open("E:/CSL/new_bst_all_80.txt", "r") as f:

        GRADE = []

        line = f.readline()
        TC_info_line = line.split()
        TC_number = int(TC_info_line[1])
        TC_count_num = int(TC_info_line[2])

        while TC_number != ((years % 100) * 100 + 1):
            for a in range(TC_count_num):
                line = f.readline()

            line = f.readline()
            TC_info_line = line.split()
            TC_number = int(TC_info_line[1])
            TC_count_num = int(TC_info_line[2])

        a = 0

        while True:

            a += 1

            if a > 1:
                line = f.readline()
                if not line:
                    break
                TC_info_line = line.split()
                TC_number = int(TC_info_line[1])
                TC_count_num = int(TC_info_line[2])

            idx = int(TC_number / 100)
            if idx < 30:
                idx += 2000
            else:
                idx += 1900

            if years == idx:
                # GRADE.append("===============================")
                # GRADE.append(["TC_count_num is",TC_count_num])
                # GRADE.append("===============================")
                for i in range(TC_count_num):
                    data = f.readline()
                    grade = int(data.split()[2])
                    GRADE.append(grade)

    return GRADE


def get_tc_date_yr(years):

    with open("E:/CSL/new_bst_all_80.txt", "r") as f:

        yr = []
        mo = []
        dy = []
        hr = []

        line = f.readline()
        TC_info_line = line.split()
        TC_number = int(TC_info_line[1])
        TC_count_num = int(TC_info_line[2])

        while TC_number != ((years % 100) * 100 + 1):
            for a in range(TC_count_num):
                line = f.readline()

            line = f.readline()
            TC_info_line = line.split()
            TC_number = int(TC_info_line[1])
            TC_count_num = int(TC_info_line[2])

        a = 0

        while True:

            a += 1

            if a > 1:
                line = f.readline()
                if not line:
                    break
                TC_info_line = line.split()
                TC_number = int(TC_info_line[1])
                TC_count_num = int(TC_info_line[2])

            idx = int(TC_number / 100)
            if idx < 30:
                idx += 2000
            else:
                idx += 1900

            if years == idx:
                # yr.append("===============================")
                # yr.append(["TC_count_num is",TC_count_num])
                # yr.append("===============================")
                # mo.append("===============================")
                # mo.append(["TC_count_num is",TC_count_num])
                # mo.append("===============================")
                # dy.append("===============================")
                # dy.append(["TC_count_num is",TC_count_num])
                # dy.append("===============================")
                # hr.append("===============================")
                # hr.append(["TC_count_num is",TC_count_num])
                # hr.append("===============================")
                for i in range(TC_count_num):
                    data = f.readline()
                    date = int(data.split()[0])
                    ayr = int(date * 0.000001)
                    amo = int(date * 0.0001) - (ayr * 100)
                    ady = int(date * 0.01) - int(date * 0.0001) * 100
                    ahr = date % 100

                    yr.append(ayr)
                    mo.append(amo)
                    dy.append(ady)
                    hr.append(ahr)
    return yr, mo, dy, hr

