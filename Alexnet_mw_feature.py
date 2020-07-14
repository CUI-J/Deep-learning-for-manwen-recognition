# -*- coding: utf-8 -*-

import tensorflow as tf
#from util import Data
from tensorflow.contrib.keras.api.keras.utils import to_categorical
from sklearn.utils import shuffle
import os
import numpy as np
from PIL import Image
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

dropout=0.5
#图片的大小,图片的长和宽都为image_size
image_size = 64
#每一次训练所使用的图片个数
batch_size = 100
#所有的数据分为多少类
num_class = 666
#原始图片所在路径
image_path = './2018new/001/001_007.jpg'
#网络模型保存的路径
model_file_path = 'Alexnetmodel/Alexnetmodel.ckpt'
#特征图保存的位置
feature_map_dir = 'Alex_fearure_map'



def inference(images):
    #定义参数
    parameters = []
    #记录每一次层的输出
    layers = {}
    #第一层卷积层
    with tf.name_scope("conv1") as scope:
        #设置卷积核11×11,3通道,64个卷积核
        kernel1 = tf.Variable(tf.truncated_normal([11,11,3,8],mean=0,stddev=0.1,
                                                  dtype=tf.float32),name="weights")
        #卷积,卷积的横向步长和竖向补偿都为4
        conv = tf.nn.conv2d(images,kernel1,[1,4,4,1],padding="SAME")
        #初始化偏置
        biases = tf.Variable(tf.constant(0,shape=[8],dtype=tf.float32),trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #RELU激活函数
        conv1 = tf.nn.relu(bias,name=scope)
        #统计参数
        parameters += [kernel1,biases]
        #lrn处理
        lrn1 = tf.nn.lrn(conv1,4,bias=1,alpha=1e-3/9,beta=0.75,name="lrn1")
        #最大池化
        pool1 = tf.nn.max_pool(lrn1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID",name="pool1")

    layers['conv1'] = conv1
    layers['pool1'] = pool1
    #第二层卷积层
    with tf.name_scope("conv2") as scope:
        #初始化权重
        kernel2 = tf.Variable(tf.truncated_normal([5,5,8,16],dtype=tf.float32,stddev=0.1)
                              ,name="weights")
        conv = tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")
        #初始化偏置
        biases = tf.Variable(tf.constant(0,dtype=tf.float32,shape=[16])
                             ,trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #RELU激活
        conv2 = tf.nn.relu(bias,name=scope)
        parameters += [kernel2,biases]
        #LRN
        lrn2 = tf.nn.lrn(conv2,4,1.0,alpha=1e-3/9,beta=0.75,name="lrn2")
        #最大池化
        pool2 = tf.nn.max_pool(lrn2,[1,3,3,1],[1,2,2,1],padding="VALID",name="pool2")

    layers['conv2'] = conv2
    #第三层卷积层
    with tf.name_scope("conv3") as scope:
        #初始化权重
        kernel3 = tf.Variable(tf.truncated_normal([3,3,16,32],dtype=tf.float32,stddev=0.1)
                              ,name="weights")
        conv = tf.nn.conv2d(pool2,kernel3,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,shape=[32],dtype=tf.float32),trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #RELU激活层
        conv3 = tf.nn.relu(bias,name=scope)
        parameters += [kernel3,biases]

    layers['conv3'] = conv3
    #第四层卷积层
    with tf.name_scope("conv4") as scope:
        #初始化权重
        kernel4 = tf.Variable(tf.truncated_normal([3,3,32,32],stddev=0.1,dtype=tf.float32),
                              name="weights")
        #卷积
        conv = tf.nn.conv2d(conv3,kernel4,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[32]),trainable=True,name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #RELU激活
        conv4 = tf.nn.relu(bias,name=scope)
        parameters += [kernel4,biases]

    layers['conv4'] = conv4
    #第五层卷积层
    with tf.name_scope("conv5") as scope:
        #初始化权重
        kernel5 = tf.Variable(tf.truncated_normal([3,3,32,16],stddev=0.1,dtype=tf.float32),
                              name="weights")
        conv = tf.nn.conv2d(conv4,kernel5,strides=[1,1,1,1],padding="SAME")
        biases = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[16]),name="biases")
        bias = tf.nn.bias_add(conv,biases)
        #REUL激活层
        conv5 = tf.nn.relu(bias)
        parameters += [kernel5,bias]
        #最大池化
        pool5 = tf.nn.max_pool(conv5,[1,3,3,1],[1,2,2,1],padding="VALID",name="pool5")

    layers['conv5'] = conv5
    #第六层全连接层
#    pool5 = pool2
    shape_pool5 = pool5.get_shape().as_list()
    reshape_num = shape_pool5[1]*shape_pool5[2]*shape_pool5[3]
    pool5 = tf.reshape(pool5,(-1,reshape_num))
    weight6 = tf.Variable(tf.truncated_normal([reshape_num,2048],stddev=0.1,dtype=tf.float32),
                           name="weight6")
    ful_bias1 = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[2048]),name="ful_bias1")
    ful_con1 = tf.nn.relu(tf.add(tf.matmul(pool5,weight6),ful_bias1))
    ful_con1_drop = tf.nn.dropout(ful_con1, dropout)

    #第七层全连接层
    weight8 = tf.Variable(tf.truncated_normal([2048,1000],stddev=0.1,dtype=tf.float32),
                          name="weight8")
    ful_bias3 = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[1000]),name="ful_bias3")
    ful_con3 = tf.nn.relu(tf.add(tf.matmul(ful_con1_drop,weight8),ful_bias3))
    ful_con3_drop = tf.nn.dropout(ful_con3, dropout)

    #softmax层
    weight9 = tf.Variable(tf.truncated_normal([1000,666],stddev=0.1),dtype=tf.float32,name="weight9")
    bias9 = tf.Variable(tf.constant(0.0,shape=[666]),dtype=tf.float32,name="bias9")
    output_softmax = tf.nn.softmax(tf.matmul(ful_con3_drop,weight9)+bias9)

    return output_softmax,parameters,layers
    
#定义所需的placeholder
keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder(dtype=tf.float32,shape=[None,image_size,image_size,3])
Y = tf.placeholder(dtype=tf.float32,shape=[None,num_class])
#构建网络获得网络的各个节点的输出
output,parameters,layers = inference(X)
#初始化
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    #载入模型
    saver.restore(sess,model_file_path)
    img_in = Image2array(image_path)
    #获得特征图
    feature_map = sess.run(layers['conv1'],feed_dict={X:img_in})
    #保存特征图
    array2Image(feature_map[0],feature_map_dir)



