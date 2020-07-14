import cv2
from math import fabs,sin,cos,radians
import numpy as np
import random
import os
import shutil

#将图片旋转一个角度
#top_img:需要旋转的图片
#degree:旋转的角度
#saveName:旋转后的图片保存的路径
def rotate(top_img,degree=5,saveName='rotate.jpg'):
    img = cv2.imread(top_img)#读取图像
    height,width=img.shape[:2]
    #旋转后的尺寸
    heightNew=int(width*fabs(sin(radians(degree)))+height*fabs(cos(radians(degree))))
    widthNew=int(height*fabs(sin(radians(degree)))+width*fabs(cos(radians(degree))))
    #得到旋转后的图片：第一个参数旋转中心，第二个参数旋转角度，第三个参数：缩放比例
    matRotation=cv2.getRotationMatrix2D((width/2,height/2),degree,1)
    matRotation[0,2] +=(widthNew-width)/2  
    matRotation[1,2] +=(heightNew-height)/2  
    # 得到一个新的图像矩阵: 原图像 ，变换矩阵，变换后尺寸，边界取值
    imgRotation=cv2.warpAffine(img,matRotation,(widthNew,heightNew),borderValue=(255,255,255))
    cv2.imwrite(saveName,imgRotation)#保存图像，第一个参数是保存的路径及文件名，第二个是图像矩阵

#图片仿射变换
def affineTransformation(img_name,size=None,post=None,mode=0,saveName=None):
    img = cv2.imread(img_name)
    rows,cols,ch = img.shape
    if size == None:
        size = (cols,rows)
    xmin = random.uniform(0,cols/2-1)
    ymin = random.uniform(0,rows/2-1)
    xmax = random.uniform(cols/2,cols)
    ymax = random.uniform(rows/2,rows)
    if mode == 0:
        if post == None:
            post = (np.float32([[50,50],[200,50],[50,200]]),
                    np.float32([[10,100],[200,50],[100,250]])
                    )
        M = cv2.getAffineTransform(post[0],post[1])
        dst = cv2.warpAffine(img,M,size,borderValue=[255,255,255])
    elif mode == 1:
        if post == None:
            post = (np.float32([[0,0],[cols,0],[0,rows],[cols,rows]]),
                    np.float32([[xmin,ymin],[xmax,ymin],[xmin,ymax],[xmax,ymax]])
                    )
        # 进行透视变换 
        # 可以先用四个点来确定一个3*3的变换矩阵
        M = cv2.getPerspectiveTransform(post[0],post[1])
        #生成3*3的矩阵图像：原图像，变换矩阵，变换后尺寸
        dst = cv2.warpPerspective(img,M,size,borderValue=(255,255,255))
    cv2.imwrite(saveName,dst)

#原来图片的位置
image_path = '../2018/'
#image_path下的所有文件夹名列表，
file_folder_list = os.listdir(image_path)
#数据增广后保存的位置
save_image_path = './2018new'
#循环文件夹名列表
for index,file_folder in enumerate(file_folder_list):
    print(index)
    #将数据所在文件夹名和文件夹的路径拼接起来
    file_folder_path = os.path.join(image_path,file_folder)
    #将数据所在文件夹名和数据增广后保存的位置路径拼接起来
    save_file_folder_path = os.path.join(save_image_path,file_folder)
    #如果save_file_folder_path路径不存在,创建这个文件夹
    if not os.path.exists(save_file_folder_path):
        os.makedirs(save_file_folder_path)
    #获取文件夹file_folder(001,002,...,666中的一个)下的所有文件名列表
    file_name_list = os.listdir(file_folder_path)
    #如果这个文件夹下的文件个数大于20个,不做数据增广,将图片复制到新的文件夹(2018new)下
    if len(file_name_list) > 20:
        for file_ in file_name_list:
            file_path = os.path.join(file_folder_path,file_)
            save_file_path = os.path.join(save_file_folder_path,file_)
            shutil.copyfile(file_path,save_file_path)
        continue
    #数据增广,旋转和仿射
    for file_ in file_name_list:
        #文件的路径
        file_path = os.path.join(file_folder_path,file_)
        #旋转
        for d in range(-5,6):
            file_s = file_[:-4] + '{}.jpg'.format(d)
            save_file_path = os.path.join(save_file_folder_path,file_s)
            rotate(file_path,degree=d,saveName=save_file_path)      
        
        #仿射变换需要用的参数
        post1 = (np.float32([[50,50],[200,50],[50,200]]),
                 np.float32([[45,50],[200,50],[60,200]]))
        post2 = (np.float32([[50,50],[200,50],[50,200]]),
                 np.float32([[50,55],[200,50],[50,210]]))
        post3 = (np.float32([[50,50],[200,50],[50,200]]),
                 np.float32([[55,50],[200,50],[45,200]]))
        post4 = (np.float32([[50,50],[200,50],[50,200]]),
                 np.float32([[50,45],[200,50],[50,190]]))
        post = [post1,post2,post3,post4]
        #仿射
        for index_j,j in enumerate(post):
            file_s = file_[:-4] + '{}fanse.jpg'.format(index_j)
            save_file_path = os.path.join(save_file_folder_path,file_s)
            affineTransformation(file_path,post=j,saveName=save_file_path)

#数据增广后的文件夹在的文件夹名列表
newfile_folder_list = os.listdir(save_image_path)
#循环文件夹名列表
for file_folder in newfile_folder_list:
    print(file_folder)
    #将数据所在文件夹名和文件夹的路径拼接起来
    file_folder_path = os.path.join(save_image_path,file_folder)
    #获取文件夹file_folder(001,002,...,666中的一个)下的所有文件名列表
    file_name_list = os.listdir(file_folder_path)
    for index,file_ in enumerate(file_name_list):
        #旧的文件名
        file_path = os.path.join(file_folder_path,file_)
        #新的文件名
        newfileName = file_folder + '_' + (3 - len(str(index)))*'0' + str(index) + '.jpg'
        newfile_path = os.path.join(file_folder_path,newfileName)
        #重命名
        os.rename(file_path,newfile_path)

#a = '../2018/001/001_001.jpg'
#rotate(a,10,saveName='-5.jpg')
#post = (np.float32([[50,50],[200,50],[50,200]]),
#        np.float32([[45,50],[200,50],[60,200]]))
#affineTransformation(a,post=post,saveName='aff.jpg')


