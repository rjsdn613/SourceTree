"""
아래는 태풍 번호별로 불러올수있는 코드

"""


def get_tc_lat(TC_numbers):

    with open("E:/CSL/new_bst_all_80.txt", "r") as f:

        while True:

            line = f.readline()

            if not line:
                break

            TC_info_line = line.split()

            TC_number = int(TC_info_line[1])
            TC_count_num = int(TC_info_line[2])

            if TC_numbers == TC_number:
                TC_data = []
                LAT = []
                for i in range(TC_count_num):
                    data = f.readline()

                    TC_data.append(data)
                    lat = int(TC_data[i].split()[3]) * 0.1
                    LAT.append(lat)
    return LAT


def get_tc_lon(TC_numbers):

    with open("E:/CSL/new_bst_all_80.txt", "r") as f:

        while True:

            line = f.readline()

            if not line:
                break

            TC_info_line = line.split()

            TC_number = int(TC_info_line[1])
            TC_count_num = int(TC_info_line[2])

            if TC_numbers == TC_number:
                TC_data = []
                LON = []
                for i in range(TC_count_num):
                    data = f.readline()

                    TC_data.append(data)
                    lon = int(TC_data[i].split()[4]) * 0.1
                    LON.append(lon)

    return LON


def get_tc_pres(TC_numbers):

    with open("E:/CSL/new_bst_all_80.txt", "r") as f:

        while True:

            line = f.readline()

            if not line:
                break

            TC_info_line = line.split()

            TC_number = int(TC_info_line[1])
            TC_count_num = int(TC_info_line[2])

            if TC_numbers == TC_number:
                TC_data = []
                PRES = []
                for i in range(TC_count_num):
                    data = f.readline()

                    TC_data.append(data)
                    pres = int(TC_data[i].split()[5])
                    PRES.append(pres)

    return PRES


def get_tc_wind(TC_numbers):

    with open("E:/CSL/new_bst_all_80.txt", "r") as f:

        while True:

            line = f.readline()

            if not line:
                break

            TC_info_line = line.split()

            TC_number = int(TC_info_line[1])
            TC_count_num = int(TC_info_line[2])

            if TC_numbers == TC_number:
                TC_data = []
                WIND = []
                for i in range(TC_count_num):
                    data = f.readline()

                    TC_data.append(data)
                    wind = int(TC_data[i].split()[6])*0.514
                    WIND.append(wind)

    return WIND


def get_tc_grade(TC_numbers):

    with open("E:/CSL/new_bst_all_80.txt", "r") as f:

        while True:

            line = f.readline()

            if not line:
                break

            TC_info_line = line.split()

            TC_number = int(TC_info_line[1])
            TC_count_num = int(TC_info_line[2])

            if TC_numbers == TC_number:
                TC_data = []
                GRADE = []
                for i in range(TC_count_num):
                    data = f.readline()

                    TC_data.append(data)
                    grade = int(TC_data[i].split()[2])
                    GRADE.append(grade)

    return GRADE


def get_tc_date(TC_numbers):

    with open("E:/CSL/new_bst_all_80.txt", "r") as f:

        while True:

            line = f.readline()

            if not line:
                break

            TC_info_line = line.split()

            TC_number = int(TC_info_line[1])
            TC_count_num = int(TC_info_line[2])

            if TC_numbers == TC_number:
                TC_data = []
                yr = []
                mo = []
                dy = []
                hr = []
                for i in range(TC_count_num):
                    data = f.readline()

                    TC_data.append(data)
                    date = int(TC_data[i].split()[0])
                    ayr = int(date * 0.000001)
                    amo = int(date * 0.0001) - (ayr * 100)
                    ady = int(date * 0.01) - int(date * 0.0001) * 100
                    ahr = date % 100

                    yr.append(ayr)
                    mo.append(amo)
                    dy.append(ady)
                    hr.append(ahr)

    return yr, mo, dy, hr


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

        idx = int(TC_number / 100)
        if idx < 30:
            idx += 2000
        else:
            idx += 1900

        while True and years == idx:

            LAT.append(["TC_count_num is", TC_count_num])
            for i in range(TC_count_num):
                data = f.readline()
                lat = int(data.split()[3]) * 0.1
                LAT.append(lat)

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

        idx = int(TC_number / 100)
        if idx < 30:
            idx += 2000
        else:
            idx += 1900

        while True and years == idx:

            LON.append(["TC_count_num is", TC_count_num])
            for i in range(TC_count_num):
                data = f.readline()
                lon = int(data.split()[3]) * 0.1
                LON.append(lon)

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

        idx = int(TC_number / 100)
        if idx < 30:
            idx += 2000
        else:
            idx += 1900

        while True and years == idx:

            PRES.append(["TC_count_num is", TC_count_num])
            for i in range(TC_count_num):
                data = f.readline()
                pres = int(data.split()[3]) * 0.1
                PRES.append(pres)

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

        idx = int(TC_number / 100)
        if idx < 30:
            idx += 2000
        else:
            idx += 1900

        while True and years == idx:

            WIND.append(["TC_count_num is", TC_count_num])
            for i in range(TC_count_num):
                data = f.readline()
                wind = int(data.split()[6])*0.514
                WIND.append(wind)

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

        idx = int(TC_number / 100)
        if idx < 30:
            idx += 2000
        else:
            idx += 1900

        while True and years == idx:

            GRADE.append(["TC_count_num is", TC_count_num])
            for i in range(TC_count_num):
                data = f.readline()
                grade = int(data.split()[3]) * 0.1
                GRADE.append(grade)

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

        idx = int(TC_number / 100)
        if idx < 30:
            idx += 2000
        else:
            idx += 1900

        while True and years == idx:

            yr.append(["TC_count_num is", TC_count_num])
            mo.append(["TC_count_num is", TC_count_num])
            dy.append(["TC_count_num is", TC_count_num])
            hr.append(["TC_count_num is", TC_count_num])

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

    return yr, mo, dy, hr

