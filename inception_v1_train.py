# -*- coding: utf-8 -*-
import inception_v1
import tensorflow as tf
import os
import numpy as np
from PIL import Image
from tensorflow.contrib import slim
from tensorflow.contrib.keras.api.keras.utils import to_categorical
from TestData import getTestSet2018WithStyle

#图片的大小,图片的长和宽都为image_size
image_size = inception_v1.inception_v1.default_image_size
#每一次训练所使用的图片个数
batch_size = 64
#所有的数据分为多少类
num_class = 666
#网络训练次数
num_epoch = 3000
#图片所在路径
#image_path = '/raid/zrr/mw2018'     #大数据
image_path = './2018new'             #数据增广后的小数据
#image_path = '../2018'              #原始小数据
#测试图片所在的所有文件夹
image_path_TestData2018 = '../TestSet2018'

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
    for file_ in file_name_list[:200]:
        file_path = os.path.join(file_folder_path,file_)
        f = Image.open(file_path)
        f = f.resize((image_size,image_size))
        arr = np.asarray(f,dtype="float32")
        X_train.append(arr)
        Y_train.append(int(file_folder)-1)
       

##################################
X_train=np.array(X_train)
Y_train=np.array(Y_train)

#print(Y_train)
X_train /=255.
#batch_X = X_train.reshape(X_train.shape[0], 64,64, 3)  # 这里是新添加的

print('X_train shape:', X_train.shape)

# one hot encode outputs  
Y_train = to_categorical(Y_train,666)
img_train,label_train,img_test,label_test = splitData(X_train,Y_train,0.2)

TestData2018_x,TestData2018_y = getTestSet2018WithStyle(image_path_TestData2018,image_size=image_size)
TestData2018_y = to_categorical(TestData2018_y,666)

X = tf.placeholder(tf.float32,[None,image_size,image_size,3])
Y = tf.placeholder(tf.float32,[None,num_class])

with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
    logits, end_points = inception_v1.inception_v1(inputs=X,num_classes=num_class,dropout_keep_prob=0.5)


cross_entropy = tf.reduce_mean(tf.losses.softmax_cross_entropy(Y,logits))
#使用优化器减小损失
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

#计算准确率
correct_prediction = tf.equal(tf.argmax(end_points['Predictions'],1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

if not os.path.exists('./inceptionv1_model'):
    os.mkdir('./inceptionv1_model')

with tf.Session() as sess:
    sess.run(init)
    for i in range(num_epoch+1):
        #获取一个batch的数据
        batch_X,batch_Y = getBatch(img_train,label_train,batch_size)
        accu,loss = sess.run([accuracy,cross_entropy],feed_dict={X:batch_X,Y:batch_Y})
        sess.run(train_step,feed_dict={X:batch_X,Y:batch_Y})
        print("step %d, train accuracy %g, train loss %g"%(i,accu,loss))
        if i%20 == 0:
            accu_,loss_ = sess.run([accuracy,cross_entropy],feed_dict={X:img_test,Y:label_test})
            print("step %d, test accuracy %g, test loss %g"%(i,accu_,loss_))
#                    print(sess.run(tf.argmax(output,1),feed_dict={X:batch_X}))
        if i == 500:
            saver.save(sess,'inceptionv1_model/inceptionv1_model500.ckpt')
    T_accu_= sess.run(accuracy,feed_dict={X:TestData2018_x,Y:TestData2018_y})
    print('Accuracy in Test Data 2018 is:{}'.format(T_accu_))
    saver.save(sess,'inceptionv1_model/inceptionv1_model1000.ckpt')

















