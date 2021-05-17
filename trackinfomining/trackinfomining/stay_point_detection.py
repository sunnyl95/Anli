# -*- coding: utf-8 -*-
import time
from math import radians, cos, sin, asin, sqrt
import pandas as pd
from get_stay_point_data import get_stay_point_data
time_format = '%Y-%m-%d,%H:%M:%S'

# 定义Point类
class Point:
    def __init__(self, latitude, longitude, dateTime, arriveTime, leaveTime):
        self.latitude = latitude
        self.longitude = longitude
        self.dateTime = dateTime
        self.arriveTime = arriveTime
        self.leaveTime = leaveTime

# 计算两经纬度坐标之间的距离
def getDistanceOfPoints(pi, pj):
    lat1, lon1, lat2, lon2 = list(map(radians, [float(pi.latitude), float(pi.longitude),
                                                float(pj.latitude), float(pj.longitude)]))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    m = 6371000 * c
    return m

# 计算两点之间的时间差,单位秒
def getTimeIntervalOfPoints(pi, pj):
    t_i = time.mktime(time.strptime(pi.dateTime, time_format))
    t_j = time.mktime(time.strptime(pj.dateTime, time_format))
    return t_j - t_i

# 计算一组坐标的平均值
def computMeanCoord(gpsPoints):
    lat = 0.0
    lon = 0.0
    for point in gpsPoints:
        lat += float(point.latitude)
        lon += float(point.longitude)
    return (lat/len(gpsPoints), lon/len(gpsPoints))

# 提取停留点
# 输入:
#        file: GPS log 文件
#        distThres: 距离阈值， 默认值200m
#        timeThres: 时间间隔阈值, 默认值30min
# according to [1]
def stay_point_extraction(points, distThres=200, timeThres=30*60):
    stayPointCenterList = []
    stayPointList = []
    pointNum = len(points)
    i = 0

    # 双层遍历
    while i < pointNum - 1:
        # 当前的距离超过距离阈值，则说明不在一个停留区域，停止进行聚合。
        j = i + 1
        flag = False
        while j < pointNum:
            if getDistanceOfPoints(points[i], points[j]) < distThres:
                j += 1
            else:
                break

        j -= 1
        # 在距离阈值范围内，至少有一个轨迹点
        if j > i:
            # 聚合候选点
            if getTimeIntervalOfPoints(points[i], points[j]) > timeThres:
                nexti = i + 1
                j += 1
                while j < pointNum:
                    if getDistanceOfPoints(points[nexti], points[j]) < distThres and \
                            getTimeIntervalOfPoints(points[nexti], points[j]) > timeThres:
                        nexti += 1
                        j += 1
                    else:
                        break
                j -= 1

                # 求均值
                latitude, longitude = computMeanCoord(points[i: j+1])
                # 获取达到时间
                arriveTime = time.mktime(time.strptime(
                    points[i].dateTime, time_format))
                # 获取离开时间
                leaveTime = time.mktime(time.strptime(
                    points[j].dateTime, time_format))
                dateTime = time.strftime(time_format, time.localtime(
                    arriveTime)), time.strftime(time_format, time.localtime(leaveTime))
                stayPointCenterList.append(
                    Point(latitude, longitude, dateTime, arriveTime, leaveTime))
                stayPointList.extend(points[i: j+1])
        i = j + 1
    return stayPointCenterList, stayPointList


# 将dataframe格式数据转为Point类型
def df_to_points(df, lat_feature, lng_feature, time_feature):

    # 判断time_feature、lat_feature、lng_feature是否为df中的特征名
    if time_feature not in list(df.columns):
        print("time feature name is not in columns! Please check feature name!")
        return

    if lat_feature not in list(df.columns):
        print("latitude feature name is not in columns! Please check feature name!")
        return

    if lng_feature not in list(df.columns):
        print("longitude feature name is not in columns! Please check feature name!")
        return

    # 判断是否有缺失值
    if df[lat_feature].isnull().sum() > 0:
        print("latitude exist NAN value!")
        return
    if df[lng_feature].isnull().sum() > 0:
        print("longitude exist NAN value!")
        return
    if df[time_feature].isnull().sum() > 0:
        print("date  exist NAN value!")
        return

    # 将date转成秒格式

    def datetosen(time_str):
        return time.mktime(time.strptime(time_str, time_format))

    # 根据时间先后顺序对样本进行排序
    df["time_sen"] = df[time_feature].apply(datetosen)
    df = df.sort_values(by="time_sen")
    df = df.drop("time_sen", axis=1)

    # 将df样本数据转成Points格式
    points = []
    for index in range(0, df.shape[0]):
        points.append(Point(df.iloc[index][lat_feature], df.iloc[index]
                            [lng_feature], df.iloc[index][time_feature], 0, 0))

    return points


def stay_point_detection(df, lat_feature, lng_feature, time_feature):

    points = df_to_points(df, lat_feature, lng_feature, time_feature)
    stayPointCenter, stayPoint = stay_point_extraction(points)

    # if exist stay point
    if len(stayPointCenter) > 0:
        lat_list = []
        lng_list = []
        arriveTime_list = []
        leaveTime_list = []
        for sp in stayPointCenter:
            lat_list.append(sp.latitude)
            lng_list.append(sp.longitude)
            arriveTime_list.append(time.strftime(
                time_format, time.localtime(sp.arriveTime)))
            leaveTime_list.append(time.strftime(
                time_format, time.localtime(sp.leaveTime)))

        result_df = pd.DataFrame({"laltitude": lat_list, "longitude": lng_list,
                                  "arriving_time": arriveTime_list, "leave_time": leaveTime_list})
        return result_df

    else:
        print("has no stay point")
        return None


if __name__ == '__main__':
    print("-----------demo-----------------")
    df = get_stay_point_data()
    result_df = stay_point_detection(df, 'lat', 'lng', 'date')
    print("stay point detect result is :\n", result_df)