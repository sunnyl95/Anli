# -*- coding: utf-8 -*-

import math
from geohash import decode
import math
from math import pi
from math import sin, cos

__author__ = 'Sun HuiLing'

# 编码字典表
__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'

# 地球半径
EARTH_REDIUS = 6378.137

# 转为弧度制表示
def rad(d):
    return d * pi / 180.0


def geohash_distance(geohash_1, geohash_2, ):
    """
    将geohash转换为经度/经度，然后计算以米为单位的两点距离。
    :param geohash_1:first geohash coordinate
    :param geohash_2:second geohash coordinate
    :return: distance between two coordinate /m
    """

    # 检查参数是否是合法的geohash编码字符
    if len([x for x in geohash_1 if x in __base32]) != len(geohash_1):
        raise ValueError('Geohash 1: %s is not a valid geohash' % (geohash_1,))

    if len([x for x in geohash_2 if x in __base32]) != len(geohash_2):
        raise ValueError('Geohash 2: %s is not a valid geohash' % (geohash_2,))


    # geohash解码：经纬度

    lat1, lng1 = decode(geohash_1)
    lat2, lng2 = decode(geohash_2)

    #打印编码结果, 检查解码是否正确
    #print(decode(geohash_1), decode(geohash_2))

    # 转为弧度制表示
    radLat1 = rad(lat1)
    radLat2 = rad(lat2)

    # 计算距离
    a = radLat1 - radLat2
    b = rad(lng1) - rad(lng2)
    distance = 2 * math.asin(math.sqrt(math.pow(sin(a / 2), 2) + cos(radLat1) * cos(radLat2) * math.pow(sin(b / 2), 2)))
    distance = distance * EARTH_REDIUS  #单位是km
    distance = distance * 1000 # km单位换算成m

    return distance