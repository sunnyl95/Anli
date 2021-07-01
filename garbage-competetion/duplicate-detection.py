#coding:utf8
import os
from PIL import Image,ImageDraw,ImageFile
import numpy
import pytesseract
import cv2
import imagehash
import collections

def compare_image_with_hash(image_file1=r"./train/images/2.jpg",
                                image_file2=r"./train/images/3.jpg", max_dif=0):
        """
        max_dif: 允许最大hash差值, 越小越精确,最小为0
        推荐使用
        """
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        hash_1 = None
        hash_2 = None
        with open(image_file1, 'rb') as fp:
            hash_1 = imagehash.average_hash(Image.open(fp))
        with open(image_file2, 'rb') as fp:
            hash_2 = imagehash.average_hash(Image.open(fp))
        dif = hash_1 - hash_2
        if dif < 0:
            dif = -dif
        if dif <= max_dif:

            return True
        else:

            return False

if __name__ == "__main__":
    root_path1 = "./train/images"
    root_path2 = "./test/images"
    for file1 in os.listdir(root_path1):
        for file2 in os.listdir(root_path2):
            if compare_image_with_hash(os.path.join(root_path1, file1), os.path.join(root_path2, file2)):
                print(os.path.join(root_path1, file1))
                print(os.path.join(root_path2, file2))
    print("Finished!")
