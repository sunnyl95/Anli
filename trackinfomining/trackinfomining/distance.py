# 计算两个经纬度距离代码：

import math
from math import pi
from math import sin, cos

EARTH_REDIUS = 6378.137  # 地球半径

# 转为弧度制表示


def rad(d):
    return d * pi / 180.0

# 计算距离
def getDistance(lat1, lng1, lat2, lng2):
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    s = 2 * math.asin(math.sqrt(math.pow(sin(a / 2), 2) +
                                cos(radLat1) * cos(radLat2) * math.pow(sin(b / 2), 2)))
    s = s * EARTH_REDIUS  # 单位是km
    s = s * 1000  # 千米  转 米
    return s


if __name__ == '__main__':
    lat1, lng1, lat2, lng2 = 1, 1, 2, 2
    d = getDistance(lat1, lng1, lat2, lng2)
    print("经纬度({lat1},{lng1})和({lat2},{lng2})之间的距离为:{distance}米".format(lat1=lat1, lng1=lng1, lat2=lat2, lng2=lng2,distance=d))
