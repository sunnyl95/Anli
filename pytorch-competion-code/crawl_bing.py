#-*-coding:utf-8-*-

import requests
import re
import os

# for i in range(40):
#   os.system("mkdir fetch\\%d"%i)

r = re.compile('src="http(.*?)"')

headers = {'User-Agent':'Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.103 Safari/537.36'}

cc = {
"0": "其他垃圾/一次性快餐盒",
"1": "其他垃圾/垃圾袋",
"2": "其他垃圾/烟头",
"3": "其他垃圾/牙签",
"4": "其他垃圾/破碗",
"5": "其他垃圾/筷子",
"6": "厨余垃圾/剩饭剩菜",
"7": "厨余垃圾/大骨头",
"8": "厨余垃圾/水果果皮",
"9": "厨余垃圾/水果果肉",
"10": "厨余垃圾/茶叶渣",
"11": "厨余垃圾/菜叶菜根",
"12": "厨余垃圾/蛋壳",
"13": "厨余垃圾/鱼刺",
"14": "可回收物/充电宝",
"15": "可回收物/背包",
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
"31": "可回收物/醋瓶",
"32": "可回收物/酒瓶",
"33": "可回收物/金属食品罐",
"34": "可回收物/锅",
"35": "可回收物/食用油桶",
"36": "可回收物/饮料瓶",
"37": "有害垃圾/干电池",
"38": "有害垃圾/软膏",
"39": "有害垃圾/过期药物"
}

for i in range(1):
    query = "可乐瓶"#cc[str(i)].split('/')[1]
    folder = query
    if not os.path.exists(folder):
        os.makedirs(folder)
    print(folder)
    c = 0
    flag = False
    first = 0
    count = 50
    max_count = 200
    query = query.replace(' ', '+')
    k = 1
    while True:
        if k > 40:
            break
        url = 'http://www.bing.com/images/async?q=%s&lostate=r&mmasync=1&first=%d&count=%d'%(query, first, count)
        res = requests.get(url, headers=headers)
        imgs = r.findall(res.text)
        for item in imgs:
            print(i, c, item)
            try:
                k += 1
                ress = requests.get('http' + item)
                ff = open(folder + '\\bing_%d.jpg'%c+200, 'wb')
                ff.write(ress.content)
                ff.close()
                c += 1
            except:
                pass
                if c == max_count:
                    flag = True
                    break
                    if flag == True:
                        break
                        first += count