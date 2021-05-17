# -*- coding: utf-8 -*-
'''目前支持base32编码'''

from math import log10

#base32编码表
__base32 = '0123456789bcdefghjkmnpqrstuvwxyz'  

#base32解码字典
__decodemap = dict()
for i in range(len(__base32)):
    __decodemap[__base32[i]] = i

def encode(latitude, longitude, precision=12):
    """
    对经纬度信息进行geohash编码
    :param latitude:经度
    :param longitude:纬度
    :param precision:精确位数， 默认是12位
    :return:纬度、经度、纬度错误浮动范围 和 纬度错误浮动范围
    """
    lat_interval = (-90.0, 90.0)
    lon_interval = (-180.0, 180.0)
    geohash = []
    bits = [16, 8, 4, 2, 1]
    bit = 0
    ch = 0
    even = True

    #开始编码指定精确位数
    while len(geohash) < precision:
        #经度
        if even:
            mid = (lon_interval[0] + lon_interval[1]) / 2
            if longitude > mid:
                ch |= bits[bit]
                lon_interval = (mid, lon_interval[1])
            else:
                lon_interval = (lon_interval[0], mid)

        #纬度
        else:
            mid = (lat_interval[0] + lat_interval[1]) / 2
            if latitude > mid:
                ch |= bits[bit]
                lat_interval = (mid, lat_interval[1])
            else:
                lat_interval = (lat_interval[0], mid)

        even = not even

        if bit < 4:
            bit += 1
        else:
            geohash += __base32[ch]
            bit = 0
            ch = 0

    return ''.join(geohash)

def decode_exactly(geohash):
    """
    对给定的geohash编码值进行解码
    :param geohash:
    :return:纬度、经度、纬度错误浮动范围 和 纬度错误浮动范围
    """
    lat_interval, lon_interval = (-90.0, 90.0), (-180.0, 180.0)
    lat_err, lon_err = 90.0, 180.0
    is_even = True
    for c in geohash:
        cd = __decodemap[c]
        for mask in [16, 8, 4, 2, 1]:
            # 经度
            if is_even:
                lon_err /= 2
                if cd & mask:
                    lon_interval = ((lon_interval[0]+lon_interval[1])/2, lon_interval[1])
                else:
                    lon_interval = (lon_interval[0], (lon_interval[0]+lon_interval[1])/2)

            #纬度
            else:
                lat_err /= 2
                if cd & mask:
                    lat_interval = ((lat_interval[0]+lat_interval[1])/2, lat_interval[1])
                else:
                    lat_interval = (lat_interval[0], (lat_interval[0]+lat_interval[1])/2)
            is_even = not is_even

    lat = (lat_interval[0] + lat_interval[1]) / 2
    lon = (lon_interval[0] + lon_interval[1]) / 2
    return lat, lon, lat_err, lon_err


def decode(geohash):
    """
    将误差考虑进去，进行解码
    :param geohash:
    :return:经度、纬度
    """

    #第一次解码
    lat, lon, lat_err, lon_err = decode_exactly(geohash)

    # 保留小数位与原始数据一致
    lats = "%.*f" % (max(1, int(round(-log10(lat_err)))) - 1, lat)
    lons = "%.*f" % (max(1, int(round(-log10(lon_err)))) - 1, lon)

    #补零
    if '.' in lats: lats = lats.rstrip('0')
    if '.' in lons: lons = lons.rstrip('0')

    return float(lats), float(lons)


