import os
from PIL import Image
import pandas as pd
from matplotlib import pyplot as plt

def drawBar(dict):
    plt.figure(figsize=(20,6))
    plt.bar(range(len(dict.keys())), [value for value in dict.values()], align='center',yerr=0.000001, )

    plt.xticks(range(len(dict.keys())), [key for key in dict.keys()])

    #设置横坐标轴的标签说明
    plt.xlabel('w/h ratio')
    #设置纵坐标轴的标签说明
    plt.ylabel('Frequency')
    #设置标题
    plt.title('w/h ratio distribution')
    #绘图
    plt.show()

root_path = "./train/labels"

classes_dict = {
    "0": "其他垃圾/一次性快餐盒",
    "1": "其他垃圾/污损塑料",
    "2": "其他垃圾/烟蒂",
    "3": "其他垃圾/牙签",
    "4": "其他垃圾/破碎花盆及碟碗",
    "5": "其他垃圾/竹筷",
    "6": "厨余垃圾/剩饭剩菜",
    "7": "厨余垃圾/大骨头",
    "8": "厨余垃圾/水果果皮",
    "9": "厨余垃圾/水果果肉",
    "10": "厨余垃圾/茶叶渣",
    "11": "厨余垃圾/菜叶菜根",
    "12": "厨余垃圾/蛋壳",
    "13": "厨余垃圾/鱼骨",
    "14": "可回收物/充电宝",
    "15": "可回收物/包",
    "16": "可回收物/化妆品瓶",
    "17": "可回收物/塑料玩具",
    "18": "可回收物/塑料碗盆",
    "19": "可回收物/塑料衣架",
    "20": "可回收物/快递纸袋",
    "21": "可回收物/插头电线",
    "22": "可回收物/旧衣服",
    "23": "可回收物/易拉罐",
    "24": "可回收物/枕头",
    "25": "可回收物/毛绒玩具",
    "26": "可回收物/洗发水瓶",
    "27": "可回收物/玻璃杯",
    "28": "可回收物/皮鞋",
    "29": "可回收物/砧板",
    "30": "可回收物/纸板箱",
    "31": "可回收物/调料瓶",
    "32": "可回收物/酒瓶",
    "33": "可回收物/金属食品罐",
    "34": "可回收物/锅",
    "35": "可回收物/食用油桶",
    "36": "可回收物/饮料瓶",
    "37": "有害垃圾/干电池",
    "38": "有害垃圾/软膏",
    "39": "有害垃圾/过期药物"
}
label_num = {str(x):0 for x in range(40)}   #0.1间隔进行统计z
for file in os.listdir(root_path):
    label_path = os.path.join(root_path, file)
    with open(label_path, "r") as f:
        label = f.read().split(",")[-1][1:]
        label_num[label] += 1

drawBar(label_num)