# -*- coding: utf-8 -*-
import tensorflow as tf
#from util import Data
from tensorflow.contrib.keras.api.keras.utils import to_categorical
from sklearn.utils import shuffle
from TestData import getTestSet2018WithStyle
import os
import numpy as np
from PIL import Image

dropout=0.5  
#图片的大小,图片的长和宽都为image_size
image_size = 64
#每一次训练所使用的图片个数
batch_size = 100
#所有的数据分为多少类
num_class = 666
#网络训练次数
num_epoch = 10000
#图片所在路径
#image_path = '/raid/zrr/mw2018'     #大数据
image_path = './2018new'             #数据增广后的小数据
#image_path = '../2018'              #原始小数据
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
#os.listdir函数返回指定的文件夹包含的文件或文件夹的名字的列表
file_folder_list = os.listdir(image_path) 

for index,file_folder in enumerate(file_folder_list):
    print(index, file_folder)
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
##################################
X_train=np.array(X_train)
Y_train=np.array(Y_train)

#print(Y_train)
X_train /=255.

print('X_train shape:', X_train.shape)

# one hot encode outputs  
Y_train = to_categorical(Y_train,666)
X_train,Y_train=shuffle(X_train,Y_train)
img_train,label_train,img_test,label_test = splitData(X_train,Y_train,0.2)


TestData2018_x,TestData2018_y = getTestSet2018WithStyle(image_path_TestData2018)
TestData2018_y = to_categorical(TestData2018_y,666)

#主函数
def run_bechmark():
    with tf.Graph().as_default():
        #定义输入图片和图片标签信息的placeholder
        keep_prob = tf.placeholder(tf.float32)
        X = tf.placeholder(dtype=tf.float32,shape=[None,image_size,image_size,3])
        Y = tf.placeholder(dtype=tf.float32,shape=[None,num_class])
        output,parameters = inference(X)
        #计算损失
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(tf.clip_by_value(output,0.001,1)),
                                                      reduction_indices=[1]))
        #使用优化器减小损失
        train_step = tf.train.AdadeltaOptimizer(learning_rate=0.2).minimize(cross_entropy)

        #计算正确率
        correct_predict = tf.equal(tf.argmax(output,1),tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_predict,tf.float32))

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        if not os.path.exists('./Alexnetmodel'):
            os.mkdir('./Alexnetmodel')
        with tf.Session() as sess:
            sess.run(init)
            for i in range(num_epoch+1):
              
                #获取一个batch的数据
                batch_X,batch_Y = getBatch(img_train,label_train,batch_size)
                sess.run(train_step,feed_dict={X:batch_X,Y:batch_Y,keep_prob: 0.5})
                _,accu,loss = sess.run([train_step,accuracy,cross_entropy],feed_dict={X:batch_X,Y:batch_Y,keep_prob: 1.0})
                print("step %d, train accuracy %g, train loss %g"%(i,accu,loss))
                if i%20 == 0:
                    accu_,loss_ = sess.run([accuracy,cross_entropy],feed_dict={X:img_test,Y:label_test,keep_prob: 1.0})
                    print("step %d, test accuracy %g, test loss %g"%(i,accu_,loss_))
            T_accu_= sess.run(accuracy,feed_dict={X:TestData2018_x,Y:TestData2018_y,keep_prob: 1.0})
            print('Accuracy in Test Data 2018 is:{}'.format(T_accu_))
            saver.save(sess,'./Alexnetmodel/Alexnetmodel.ckpt')


def inference(images):
    #定义参数
    parameters = []
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

    #第七层第二层全连接层
 #   weight7 = tf.Variable(tf.truncated_normal([2048,2048],stddev=0.1,dtype=tf.float32),
 #                         name="weight7")
 #   ful_bias2 = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[2048]),name="ful_bias2")
 #   ful_con2 = tf.nn.relu(tf.add(tf.matmul(ful_con1_drop,weight7),ful_bias2))
 #   ful_con2_drop = tf.nn.dropout(ful_con2, dropout)

    #第八层第三层全连接层
    weight8 = tf.Variable(tf.truncated_normal([2048,1000],stddev=0.1,dtype=tf.float32),
                          name="weight8")
    ful_bias3 = tf.Variable(tf.constant(0.0,dtype=tf.float32,shape=[1000]),name="ful_bias3")
    ful_con3 = tf.nn.relu(tf.add(tf.matmul(ful_con1_drop,weight8),ful_bias3))
    ful_con3_drop = tf.nn.dropout(ful_con3, dropout)

    #softmax层
    weight9 = tf.Variable(tf.truncated_normal([1000,666],stddev=0.1),dtype=tf.float32,name="weight9")
    bias9 = tf.Variable(tf.constant(0.0,shape=[666]),dtype=tf.float32,name="bias9")
    output_softmax = tf.nn.softmax(tf.matmul(ful_con3_drop,weight9)+bias9)

    return output_softmax,parameters


if __name__ == "__main__":
    run_bechmark()

