# -*- coding: utf-8 -*-

import inception_v1
import tensorflow as tf
import os
import numpy as np
from PIL import Image
from tensorflow.contrib import slim
from tensorflow.contrib.keras.api.keras.utils import to_categorical
import matplotlib.pyplot as plt
import cv2


def array2Image(array,save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i in range(array.shape[2]):
        save_name = os.path.join(save_dir,'{}.png'.format(i))
        arr = cv2.resize(array[:,:,i],(64,64))
        plt.imsave(save_name,arr)

def Image2array(Image_name):
    f = Image.open(Image_name)
    f = f.resize((image_size,image_size))
    arr = np.asarray(f,dtype="float32")
    arr = arr[np.newaxis,:,:,:]
    return arr

#图片的大小,图片的长和宽都为image_size
image_size = inception_v1.inception_v1.default_image_size
#每一次训练所使用的图片个数
batch_size = 128
#所有的数据分为多少类
num_class = 666
#网络训练次数
num_epoch = 1000
#原始图片所在路径
image_path = './2018new1/001/001_007.jpg'
#网络模型保存的路径
model_file_path = 'inceptionv1_model/inceptionv1_model500.ckpt'
#特征图保存的位置
feature_map_dir = 'inceptionV1_fearure_map'




X = tf.placeholder(tf.float32,[None,image_size,image_size,3])
Y = tf.placeholder(tf.float32,[None,num_class])

with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
    logits, end_points = inception_v1.inception_v1(inputs=X,num_classes=num_class,is_training=False)


#cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(Y,logits))
##使用优化器减小损失
#train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)
#
##计算准确率
#correct_prediction = tf.equal(tf.argmax(end_points['Predictions'],1), tf.argmax(Y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    #载入模型
    saver.restore(sess,model_file_path)
    img_in = Image2array(image_path)
    #获得特征图
    feature_map = sess.run(end_points['Conv2d_2b_1x1'],feed_dict={X:img_in})
    #保存特征图
    array2Image(feature_map[0],feature_map_dir)

