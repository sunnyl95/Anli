'''
若该天所有的基站数据里，超过90%的基站定位都在家中，则认为是该天未出门。
若连续10天里有8天及以上的天数都未出门，则认为该吸毒人员是闭门不出的。
'''

import pandas as pd
from get_home_lat_lng import get_between_days_data
from get_home_lat_lng import get_home_lat_lng
from distance import getDistance
from get_stay_home_data import get_stay_home_data

def stay_home_detection(df, time_feature='time', lat_feature='lat', lng_feature='lng', open_day='2021-04-07',
                        close_day='2021-05-06', start_hour='00:00', end_hour='06:00', open_day_10='2021-04-26',
                        close_day_10='2021-05-06'):
    '''
    检测最近一段时间是否一直在家
    :param df: 数据表 dataframe（该人员所有记录的轨迹数据）
    :param time_feature: 表示时间的特征列名
    :param lat_feature: 表示经度的特征列名
    :param lng_feature: 表示纬度的特征列名
    :param open_day: 30天开始日期
    :param close_day: 30天截止日期
    :param start_hour: 开始时间
    :param end_hour: 截止时间
    :param open_day_10: 10天内开始日期
    :param close_day_10:10天内截止日期
    :return: 返回该人的家庭住址经纬度坐标
    '''

    # 第一步：获取家庭地址的经纬度
    home_lat, home_lng = get_home_lat_lng(df, time_feature=time_feature, lat_feature=lat_feature,
                                          lng_feature=lng_feature,
                                          open_day=open_day, close_day=close_day, start_hour=start_hour,
                                          end_hour=end_hour)

    # 第二步：获取10天内的轨迹数据，计算每一条轨迹与家的距离
    df1 = get_between_days_data(df, time_feature=time_feature, open_day=open_day_10, close_day=close_day_10)

    df1_copy = df1.copy()

    df1_copy['distance'] = df1_copy.apply(
        lambda x: getDistance(lat1=home_lat, lng1=home_lng, lat2=x["lat"], lng2=x["lng"]), axis=1)

    #第三步：判断这10天里的距离是否有90%以上的轨迹记录距离家都小于1000米  yes->闭门不出 no->非闭门不出
    rate = len(df1_copy[df1_copy["distance"] < 1000]) / len(df1_copy)
    if rate >= 0.9 :
        return True
    else:
        return False


if __name__ == '__main__':
    df = get_stay_home_data()
    result = stay_home_detection(df)
    print("近期是否闭门不出：",result)