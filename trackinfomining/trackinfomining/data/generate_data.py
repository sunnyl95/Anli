import random
import math
import time
import pandas as pd


def generate_random_gps(base_log=None, base_lat=None, radius=None):
    radius_in_degrees = radius / 111300
    u = float(random.uniform(0.0, 1.0))
    v = float(random.uniform(0.0, 1.0))
    w = radius_in_degrees * math.sqrt(u)
    t = 2 * math.pi * v
    x = w * math.cos(t)
    y = w * math.sin(t)
    longitude = y + base_log
    latitude = x + base_lat
    # 保留6位小数点
    loga = '%.6f' % longitude
    lata = '%.6f' % latitude
    return loga, lata


def generate_random_time(start_time=(2021, 1, 1, 0, 0, 0, 0, 0, 0), end_time=(2021, 5, 6, 23, 59, 59, 0, 0, 0)):
    #start = (2021, 1, 1, 0, 0, 0, 0, 0, 0)  # 设置开始日期时间元组(2021-1-1 00：00：00)
    #end = (2021, 5, 6, 23, 59, 59, 0, 0, 0)  # 设置结束日期时间元组(2021-5-6 23：59：59)

    start = time.mktime(start_time)  # 生成开始时间戳
    end = time.mktime(end_time)  # 生成结束时间戳

    # 随机生成一个日期字符串
    t = random.randint(start, end)  # 在开始和结束时间戳中随机取出一个
    date_touple = time.localtime(t)  # 将时间戳生成时间元组
    date = time.strftime("%Y-%m-%d %H:%M:%S", date_touple)  # 将时间元组转成格式化字符串(1976-05-21)

    return date


if __name__ == '__main__':
    # 时间戳起始时间设置
    start = (2021, 1, 1, 0, 0, 0, 0, 0, 0)  # 设置开始日期时间元组(2021-1-1 00：00：00)
    end = (2021, 5, 6, 23, 59, 59, 0, 0, 0)  # 设置结束日期时间元组(2021-5-6 23：59：59)

    # 经纬度参数设置
    base_lat = 103.49
    base_lng = 36.03
    base_radius = 50000  # 半径：单位是米

    lat_list = []
    lng_list = []
    time_list = []
    # 一个人产生60天的数据， 每天有100条记录
    for i in range(0, 60):
        for j in range(0, 100):
            lng, lat = generate_random_gps(base_log=base_lng, base_lat=base_lat,radius=base_radius)  # 103.49, 36.03为兰州的中心位置,100000为半径
            lat_list.append(lat)
            lng_list.append(lng)

            random_time = generate_random_time(start_time=start, end_time=end)
            time_list.append(random_time)

    #将数据写入csv
    df = pd.DataFrame({"lat": lat_list, "lng":lng_list, "time":time_list})
    df.to_csv("generate_data.csv", index=False)
    print("generate data successful!!")
