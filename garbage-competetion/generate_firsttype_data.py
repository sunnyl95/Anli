import os
import pandas as pd
from shutil import copyfile
from matplotlib import pyplot as plt
images_path = r"./train/images"
labels_path = r"./train/labels"
images_list = os.listdir(images_path)
labels_list = os.listdir(labels_path)

first_type_path = "./firstType"
if not os.path.exists(first_type_path):
    os.makedirs(first_type_path)

first_train_path = "./firstType/train"
if not os.path.exists(first_train_path):
    os.makedirs(first_train_path)

num = 0
for label_txt in labels_list:
    print(num)
    with open(os.path.join(labels_path, label_txt)) as f:
        contenct = f.read()
        label = contenct.split(",")[-1][1:]
        tmp_path = os.path.join(first_train_path, label)
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        copyfile(os.path.join(images_path, label_txt.replace("txt", 'jpg')), os.path.join(tmp_path, label_txt.replace("txt", 'jpg')))
        num += 1


images_path = r"./test/images"
labels_path = r"./test/labels"
images_list = os.listdir(images_path)
labels_list = os.listdir(labels_path)

first_test_path = "./firstType/test"
if not os.path.exists(first_test_path):
    os.makedirs(first_test_path)

num = 0
for label_txt in labels_list:
    print(num)
    with open(os.path.join(labels_path, label_txt)) as f:
        contenct = f.read()
        label = contenct.split(",")[-1][1:]
        tmp_path = os.path.join(first_test_path, label)
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)
        copyfile(os.path.join(images_path, label_txt.replace("txt", 'jpg')), os.path.join(tmp_path, label_txt.replace("txt", 'jpg')))
        num += 1
