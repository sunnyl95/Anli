"""
方案1:对吸毒人员三个月内，每天凌晨00：00-6:00之间的基站数据进行频繁地点统计，认为最频繁的地点即为吸毒人员家庭地址纬度（采用geohash编码后进行频繁统计，然后再映射回对应的经纬度）
"""

import pandas as pd
from .get_stay_home_data import *

#---------方案1------------

def get_between_days_data(df, time_feature, open_day = '2021-04-07', close_day = '2021-05-06'):
    '''
    获取指定日期之间的数据， 例如：获取2021-04-07到2021-05-06的数据
    :param df: 数据表 dataframe
    :param time_feature: 表示时间戳的特征列名，该特征类型指定是pandas时间戳类型
    :param open_day: 开始日期， 示例：'2021-04-07'
    :param close_day: 截止日期，示例：'2021-05-06'
    :return: 返回在开始时间和截止时间之间的数据，dataframe类型
    '''

    con1 = df[time_feature]>=open_day
    con2 = df[time_feature]<close_day
    between_days_df = df[con1&con2]
    return between_days_df

def get_between_hour_data(df,  time_feature, start_hour = "00:00:00", end_hour="06:00:00"):
    '''
    获取指定时间段内的数据
    :param df: 数据表 dataframe
    :param time_feature: 表示时间的特征列名，该特征类型指定是pandas时间戳类型
    :param start_hour: 开始时间， 示例："00:00:00"
    :param end_hour: 截止时间， 示例："06:00:00"
    :return: 返回在开始时间和截止时间之间的数据，dataframe类型
    '''
    return df.set_index(time_feature).between_time(start_hour, end_hour)


def get_home_lat_lng(df, time_feature='time', lat_feature = 'lat', lng_feature = 'lng' , open_day = '2021-04-07', close_day = '2021-05-06', start_hour = '00:00:00', end_hour='06:00:00'):
    '''
    获取家庭住址的经纬度坐标
    :param df: 数据表 dataframe
    :param time_feature: 表示时间的特征列名
    :param lat_feature: 表示经度的特征列名
    :param lng_feature: 表示纬度的特征列名
    :param open_day: 开始日期
    :param close_day: 截止日期
    :param start_hour: 开始时间
    :param end_hour: 截止时间
    :return: 返回该人的家庭住址经纬度坐标
    '''

    #判断time_feature、lat_feature、lng_feature是否为df中的特征名
    if time_feature not in list(df.columns):
        print("time feature name is not in columns! Please check feature name!")
        return

    if lat_feature not in list(df.columns):
        print("latitude feature name is not in columns! Please check feature name!")
        return

    if lng_feature not in list(df.columns):
        print("longitude feature name is not in columns! Please check feature name!")
        return

    #转化为pandas时间戳格式
    df[time_feature] = pd.to_datetime(df[time_feature])

    #第一步：获取指定瞄点时间内的30天的数据
    df1 = get_between_days_data(df=df, time_feature = time_feature,open_day = open_day, close_day = close_day)

    #第二步：获取这30天内00:00:00-06:00:00之间的数据
    df2 = get_between_hour_data(df=df1, time_feature = time_feature, start_hour=start_hour, end_hour = end_hour)

    #第三步：计算经纬度的平均值
    home_lat = df[lat_feature].mean()
    home_lng = df[lng_feature].mean()

    return  home_lat, home_lng

#---------------------方案1 End---------------------
