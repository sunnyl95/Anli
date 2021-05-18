# trackinfomining

trackinfomining(trajectory information mining)是针对轨迹信息挖掘而开发的工具包，主要是通过GSGA项目建模过程进行总结开发出来的一些功能组件。目的是在“数据交易沙箱”平台上供内部人员和平台采购人员使用，提高特征构建效率。（持续优化更新）

trackinfomining组件大体上包括以下功能：
- geohash编码、解码
- 计算geohash编码后的坐标距离
- 计算两经纬度坐标间的距离
- 获取指定时间段内的轨迹数据
- 获取指定日期段内的轨迹数据
- 定位居住地坐标
- 闭门不出检测
- 停留点检测


## 安装
- Install the release version of `trackinfomining` from [PYPI](https://github.com/sunnyl95/Anli/tree/main/trackinfomining/) with:
```
pip install -i https://test.pypi.org/simple/ trackinfomining
```


## 示例
This is a basic example which shows you how to use trackinfomining:
``` python
import trackinfomining as tfm

'''
trackinfomining（轨迹信息挖掘）工具包的使用手册
'''

'''
1.第一部分：示例trackinfomining组件的geohash编码功能
  geohash编码、解码、geohash编码后的两点间距离
'''
# 定义经纬度
lat1, lng1 = 1, 1
# geohash编码
encode_result1 = tfm.geohash_encode(lat1, lng1, precision=5)
print("经纬度({lat},{lng})经过5位geohash编码的结果为：{encode_result}".format(
    lat=lat1, lng=lng1, encode_result=encode_result1))
##经纬度(1,1)经过5位geohash编码的结果为：s00tw

#解码
decode_result1 = tfm.geohash_decode(encode_result1)
print("{encode_result}解码结果为：{decode_result}".format(
    encode_result=encode_result1, decode_result=decode_result1))
##s00tw解码结果为：(1.0, 1.0)

lat2, lng2 = 2, 2
#geohash编码
encode_result2 = tfm.geohash_encode(lat2, lng2, precision=5)

# 求geohash编码后的两个地点之间的距离
distances = tfm.geohash_distance(encode_result1, encode_result2)
print("\n{first_place}和{second_place}之间的距离为:{distances}米".format(
    first_place=encode_result1, second_place=encode_result2, distances=distances))
##s00tw和s037m之间的距离为:157401.56104583552米

# 求两个经纬度坐标之间的距离
d = tfm.getDistance(lat1, lng1, lat2, lng2)
print("经纬度({lat1},{lng1})和({lat2},{lng2})之间的距离为:{distance}米".format(lat1=lat1, lng1=lng1, lat2=lat2, lng2=lng2,distance=d))
##经纬度(1,1)和(2,2)之间的距离为:157401.56104583552米


'''
2.第二部分：示例trackinfomining组件的闭门不出检测功能
'''
#加载数据集
df =  tfm.get_stay_home_data()  #一个人的轨迹数据，包括经纬度和时间（"lat", "lng", "time"）
print(df.head(2))
#         lat        lng                 time
#0  103.806342  35.830415  2021-03-26 02:39:14
#1  103.823400  35.768967  2021-02-04 16:02:17

#获取df中open_day至close_day日期段的所有样本
#print(tfm.get_between_days_data(df,"time",open_day='2021-04-07', close_day='2021-05-06'))

#获取df中start_hour至end_hour时间段的所有样本
#print(tfm.get_between_hour_data(df,"time",start_hour=='00:00:00', end_hour="06:00:00"))

#获取一个人的近期居住地经纬度坐标
home_lat, home_lng = tfm.get_home_lat_lng(df, "time","lat", "lng", open_day="2021-04-07", close_day="2021-05-06", start_hour="00:00:00", end_hour="06:00:00")
print("home latitude is:", home_lat)
print("home longitude is:", home_lng)
#home latitude is: 103.49275263983333
#home longitude is: 36.02927143583333

#检测近期是否闭门不出
''' 
    df: 数据表 dataframe（该人员所有记录的轨迹数据）
    time_feature: 表示时间的特征列名
    lat_feature: 表示经度的特征列名
    lng_feature: 表示纬度的特征列名
    open_day: 用于定位居住地点的开始日期
    close_day: 用于定位居住地点的截止日期
    start_hour: 用于定位居住地点的开始时间点
    end_hour: 用于定位居住地点的截止时间点
    open_day_10: 用于确定近期是否居家未出的开始日期
    close_day_10:用于确定近期是否居家未出的截止日期
'''
result = tfm.stay_home_detection(df, time_feature='time', lat_feature='lat', lng_feature='lng',open_day='2021-04-07', close_day='2021-05-06', start_hour='00:00:00', end_hour='06:00:00',open_day_10='2021-04-26', close_day_10='2021-05-06')
#print result
print("近期是否闭门不出：",result)
#近期是否闭门不出： False


'''
3.第三部分：示例trackinfomining组件的停留点检测功能
'''
#加载数据集
df = tfm.get_stay_point_data()   #一个人的轨迹数据，包括经纬度和时间（"lat", "lng", "date"）
df.head(2)
#lat	lng	date
#0	39.984563	116.317517	2008-10-23,02:53:40
#1	39.984608	116.317761	2008-10-23,03:53:35

#获取停留点：停留点中心经纬度坐标，达到时间、离开时间
result_df = tfm.stay_point_detection(df, 'lat', 'lng', 'date')
print("stay point detect result is :\n", result_df)
# stay point detect result is :
#     laltitude   longitude        arriving_time           leave_time
# 0  39.984624  116.317798  2008-10-23,02:13:20  2008-10-23,03:53:35
# 1  40.001483  116.312620  2009-04-03,01:17:22  2009-04-03,01:50:28
```
