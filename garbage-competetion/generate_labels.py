import os
import pandas as pd
from matplotlib import pyplot as plt
images_path = r"./train/images"
labels_path = r"./train/labels"
images_list = os.listdir(images_path)
labels_list = os.listdir(labels_path)

images_path_list = [("images/"+ x) for x in images_list]
labels_path_list = []

for images_name in images_list:
    label_name = images_name.replace("jpg", "txt")
    if label_name in labels_list:
        labels_path_list.append(("labels/"+ label_name ))
    else:
        print("存在不匹配的数据{}与标签{}".format(images_name, label_name))
        pass


df = pd.DataFrame()
print(df)
df["index"] = images_path_list
df["labels"] = labels_path_list
print(df.head(5))
df.to_csv("./train/labels.csv", index=False)

