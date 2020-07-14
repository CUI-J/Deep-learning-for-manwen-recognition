# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.keras.api.keras.utils import to_categorical
import numpy as np
import os
from PIL import Image
from sklearn.utils import shuffle
from TestData import getTestSet2018WithStyle

#图片的大小,图片的长和宽都为image_size
image_size = 64
#每一次训练所使用的图片个数
batch_size = 64
#所有的数据分为多少类
num_class = 666
#网络训练次数
num_epoch = 3000
#图片所在路径
#image_path = '/raid/zrr/mw2018'     #大数据
image_path = '../2018new'             #数据增广后的小数据
#image_path = '../2018'              #原始小数据
image_path_TestData2018 = './TestSet2018'


#将数据分为训练集和测试集        
def splitData(img,label,rate):
    length = len(img)
    index = [i for i in range(length)]
    np.random.shuffle(index)
    img = img[index]
    label = label[index]
    num = int(length*rate)
    img_train = img[num:]
    img_test = img[:num]
    label_train = label[num:]
    label_test = label[:num]
    return (img_train,label_train,img_test,label_test)

index_ = 0

#获取一个batch的数据
def getBatch(img,label,batch_size):
    global index_
    if index_ + 1> len(img)//batch_size:
        index_ = 0
    start = batch_size*index_
    end = start + batch_size
    index_ = index_ + 1
    return img[start:end],label[start:end]


X_train = []
Y_train = []

file_folder_list = os.listdir(image_path)


for index,file_folder in enumerate(file_folder_list):
    print(index)
    #将数据所在文件夹名和文件夹的路径拼接起来
    file_folder_path = os.path.join(image_path,file_folder)
    file_name_list = os.listdir(file_folder_path)
    for file_ in file_name_list:
        file_path = os.path.join(file_folder_path,file_)
        f = Image.open(file_path)
        f = f.resize((image_size,image_size))
        arr = np.asarray(f,dtype="float32")
        X_train.append(arr)
        Y_train.append(int(file_folder)-1)
       
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
    conv1_1 = conv_op(input_op,name='conv1_1',kh=3,kw=3,n_out=32,dh=1,dw=1,p=p)
    conv1_2 = conv_op(conv1_1,name='conv1_2',kh=3,kw=3,n_out=32,dh=1,dw=1,p=p)
    pool1 = mpool_op(conv1_2,name='pool1',kh=2,kw=2,dw=2,dh=2)

    conv2_1 = conv_op(pool1,name='conv2_1',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
    conv2_2 = conv_op(conv2_1,name='conv2_2',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
    pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, dw=2, dh=2)

    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=128, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dw=2, dh=2)

    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dw=2, dh=2)

    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, dw=2, dh=2)

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
    return predictions,softmax,fc8,p
        
        
##################################
X_train=np.array(X_train)
Y_train=np.array(Y_train)

#X_train /=255.

print('X_train shape:', X_train.shape)

# one hot encode outputs  
Y_train = to_categorical(Y_train,666)
img_train,label_train,img_test,label_test = splitData(X_train,Y_train,0.1)

TestData2018_x,TestData2018_y = getTestSet2018WithStyle(image_path_TestData2018)
TestData2018_y = to_categorical(TestData2018_y,666)

input_shape = (64, 64, 3)

x = tf.placeholder(tf.float32, [None, 64,64,3])
y_ = tf.placeholder(tf.float32, [None, 666])
keep_prob = tf.placeholder(tf.float32)

predictions,softmax,fc8,p = inference_op(x,keep_prob)

#计算损失
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_*tf.log(tf.clip_by_value(softmax,0.001,1)),
#                                              reduction_indices=[1]))
cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_,logits=fc8))
#使用优化器减小损失
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

#计算正确率
correct_predict = tf.equal(tf.argmax(y_,1),tf.argmax(softmax,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

if not os.path.exists('./vggmodel'):
    os.mkdir('./vggmodel')

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_epoch+1):
        #获取一个batch的数据
        batch_X,batch_Y = getBatch(img_train,label_train,128)
        accu,loss = sess.run([accuracy,cross_entropy],feed_dict={x:batch_X,y_:batch_Y,keep_prob:1.0})
        sess.run(train_step,feed_dict={x:batch_X,y_:batch_Y,keep_prob:0.5})
        print("step %d, train accuracy %g, train loss %g"%(i,accu,loss))
        if i%20 == 0:
            accu_,loss_ = sess.run([accuracy,cross_entropy],feed_dict={x:img_test,y_:label_test,keep_prob:1.0})
            print("step %d, test accuracy %g, test loss %g"%(i,accu_,loss_))
#                    print(sess.run(tf.argmax(output,1),feed_dict={X:batch_X}))
    T_accu_= sess.run(accuracy,feed_dict={x:TestData2018_x,y_:TestData2018_y,keep_prob: 1.0})
    print('Accuracy in Test Data 2018 is:{}'.format(T_accu_))
    saver.save(sess,'vggmodel/vggmodel16.ckpt')
    





