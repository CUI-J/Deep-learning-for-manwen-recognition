# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.contrib.keras.api.keras.utils import to_categorical
import numpy as np
import os
from PIL import Image
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import cv2

#图片的大小,图片的长和宽都为image_size
image_size = 64
#每一次训练所使用的图片个数
batch_size = 64
#所有的数据分为多少类
num_class = 666
#网络训练次数
num_epoch = 1000
#原始图片所在路径
image_path = './2018new/002/002_007.jpg'
#网络模型保存的路径
model_file_path = 'vggmodel/vggmodel16.ckpt'
#特征图保存的位置
feature_map_dir = 'vgg16_fearure_map'

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
       
# 用来创建卷积层并把本层的参数存入参数列表
# input_op:输入的tensor name:该层的名称 kh:卷积层的高 kw:卷积层的宽 n_out:输出通道数，dh:步长的高 dw:步长的宽，p是参数列表
def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    # 输入的通道数
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",shape=[kh,kw,n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1,dh,dw,1),padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out],dtype=tf.float32)
        biases = tf.Variable(bias_init_val , trainable=True , name='b')
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z,name=scope)
        p += [kernel,biases]
        return activation

# 定义全连接层
def fc_op(input_op,name,n_out,p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
        # tf.nn.relu_layer()用来对输入变量input_op与kernel做乘法并且加上偏置b
        activation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)
        p += [kernel,biases]
        return activation
        
def fc_op1(input_op,name,n_out,p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
        # tf.nn.relu_layer()用来对输入变量input_op与kernel做乘法并且加上偏置b
#        activation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)
        out = tf.matmul(input_op,kernel) + biases
        p += [kernel,biases]
        return out

# 定义最大池化层
def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)

#定义网络结构Vgg16
def inference_op(input_op,keep_prob):
    p = []
    layers = {}
    conv1_1 = conv_op(input_op,name='conv1_1',kh=3,kw=3,n_out=32,dh=1,dw=1,p=p)
    conv1_2 = conv_op(conv1_1,name='conv1_2',kh=3,kw=3,n_out=32,dh=1,dw=1,p=p)
    pool1 = mpool_op(conv1_2,name='pool1',kh=2,kw=2,dw=2,dh=2)
    layers['conv1_1'] = conv1_1
    layers['conv1_2'] = conv1_2

    conv2_1 = conv_op(pool1,name='conv2_1',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
    conv2_2 = conv_op(conv2_1,name='conv2_2',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
    pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, dw=2, dh=2)
    layers['conv2_1'] = conv2_1
    layers['conv2_2'] = conv2_2

    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dw=2, dh=2)
    layers['conv3_1'] = conv3_1
    layers['conv3_2'] = conv3_2
    layers['conv3_3'] = conv3_3

    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dw=2, dh=2)
    layers['conv4_1'] = conv4_1
    layers['conv4_2'] = conv4_2
    layers['conv4_3'] = conv4_3

    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, dw=2, dh=2)
    layers['conv5_1'] = conv5_1
    layers['conv5_2'] = conv5_2
    layers['conv5_3'] = conv5_3

    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5,[-1,flattened_shape],name="resh1")

    fc6 = fc_op(resh1,name="fc6",n_out=4096,p=p)
    fc6_drop = tf.nn.dropout(fc6,keep_prob,name='fc6_drop')
    fc7 = fc_op(fc6_drop,name="fc7",n_out=2048,p=p)
    fc7_drop = tf.nn.dropout(fc7,keep_prob,name="fc7_drop")
    fc8 = fc_op1(fc7_drop,name="fc8",n_out=666,p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax,1)
    return predictions,softmax,fc8,p,layers
        

input_shape = (64, 64, 3)

##tensorflow实现VGG11
#
x = tf.placeholder(tf.float32, [None, 64,64,3])
y_ = tf.placeholder(tf.float32, [None, 666])
keep_prob = tf.placeholder(tf.float32)

predictions,softmax,fc8,p,layers = inference_op(x,keep_prob)


##计算正确率
#correct_predict = tf.equal(tf.argmax(y_,1),tf.argmax(softmax,1))
#accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    #载入模型
    saver.restore(sess,model_file_path)
    img_in = Image2array(image_path)
    #获得特征图
    feature_map = sess.run(layers['conv4_1'],feed_dict={x:img_in})
    #保存特征图
    array2Image(feature_map[0],feature_map_dir)

