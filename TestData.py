# -*- coding: utf-8 -*-

import os
from PIL import Image
import numpy as np

#图片的大小,图片的长和宽都为image_size
#image_size = 64
#图片所在路径
#image_path = './TestSet2018'                    #测试数据

def getTestSet2018(image_path,image_size = 64):
    file_folder_list = os.listdir(image_path)
    TestData_X = []
    TestData_Y = []
    
    for index,file_folder in enumerate(file_folder_list):
        print(index)
        #将数据所在文件夹名和文件夹的路径拼接起来
        file_folder_path = os.path.join(image_path,file_folder)
        if os.path.isdir(file_folder_path):
            file_name_list = os.listdir(file_folder_path)
        else:
            continue
        for file_ in file_name_list:
            file_path = os.path.join(file_folder_path,file_)
            f = Image.open(file_path)
            f = f.resize((image_size,image_size))
            arr = np.asarray(f,dtype="float32")
            TestData_X.append(arr)
            TestData_Y.append(int(file_folder[1:])-1)
    return np.array(TestData_X),np.array(TestData_Y)
    
#根据给的style读取相应的图片,style取值从1到11,代表11种不同的字体的图片
#image_path:图片的路径
def getTestSet2018WithStyle(image_path,image_size = 64,Style=5):
    file_folder_list = os.listdir(image_path)
    TestData_X = []
    TestData_Y = []
    Style = (2-len(str(Style)))*'0' + str(Style)
    for index,file_folder in enumerate(file_folder_list):
        print(index)
        #将数据所在文件夹名和文件夹的路径拼接起来
        file_folder_path = os.path.join(image_path,file_folder)
        if os.path.isdir(file_folder_path):
            file_name = os.path.join(file_folder_path,Style + '.jpg')
        else:
            continue
        f = Image.open(file_name)
        f = f.resize((image_size,image_size))
        arr = np.asarray(f,dtype="float32")
        TestData_X.append(arr)
        TestData_Y.append(int(file_folder[1:])-1)
    return np.array(TestData_X),np.array(TestData_Y)


if __name__ == "__main__":
    image_path = './TestSet2018'
    X,Y = getTestSet2018WithStyle(image_path)