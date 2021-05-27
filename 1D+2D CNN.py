# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:24:15 2019

@author: Renyu Liu
"""
import tensorflow as tf
import numpy as np
import os
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy import signal###################
from sklearn import preprocessing##########
from python_speech_features import mfcc,delta
from scipy import interpolate

LEARNING_RATE_BASE = 0.0001  # 最初学习率
LEARNING_RATE_DECAY = 0.9  # 学习率的衰减率
LEARNING_RATE_STEP = 300  # 喂入多少轮BATCH-SIZE以后，更新一次学习率。一般为总样本数量/BATCH_SIZE 

def get_wav(audio):
    wav = []
    n = len(audio)
    if n < 4800:
        y = interpolate.interp1d(range(n), audio[0:n], kind='cubic')
        xint = np.linspace(0,n-1,4800)
        wav_same_length = y(xint)            
        for j in range(len(wav_same_length)):
            wav.append(wav_same_length[j])
    elif n > 4800:#############这里有个大问题，直接裁剪，会把边边的给去掉
        nn = np.int32((n - 4800)/2)
        wav_same_length = audio[0+nn:n-nn]
        for j in range(len(wav_same_length)):
            wav.append(wav_same_length[j])
    else:
        for j in range(n):
            wav.append(audio[j])
    return np.array(wav[0:int(len(wav)/4800)*4800])#????????????

def gen_wavlist(wavpath):
	for (dirpath, dirnames, filenames) in os.walk(wavpath):
		lis = list(range(len(filenames)))
		wavdict = np.zeros((len(filenames),4800))
		labeldict = np.zeros((len(filenames),2))
		t1 = 0
		t2 = 0
		for i in range(len(lis)):
			j = lis[i]
			filename = filenames[j]
			if filename.endswith('.wav'):
				filepath = os.sep.join([dirpath, filename])
				fileid = filename.strip('.wav')
				fs, audio = wavfile.read(filepath)
				audio =  get_wav(audio)
#				b, a = signal.butter(6, [110/8000,1000/8000], 'bandpass')  #配置滤波器 8 表示滤波器的阶数
#				audio = signal.filtfilt(b, a, audio) #data为要过滤的信号                 
				wavdict[i,:] = audio/max(abs(audio))##可以换一下其他的
				label = fileid.split('-')[1]
				if label[0] == '0':#完整
				    labeldict[i,:] = [0,1]
				    t1 = t1 +1
				else:#不完整
				    labeldict[i,:] = [1,0] 
				    t2 = t2 +1                                 
	return wavdict, labeldict, fs, t1, t2

def mfcc_delta1_delta2(features_mfcc, delta1, delta2):
	features_mfcc_delta1_delta2 = np.zeros((3*len(features_mfcc),59,26))#####mfcc
	for i in range(len(features_mfcc)):
		features_mfcc_delta1_delta2[i*3]=features_mfcc[i]#####mfcc
		features_mfcc_delta1_delta2[i*3+1]=delta1[i]#####mfcc
		features_mfcc_delta1_delta2[i*3+2]=delta2[i] #####mfcc  
	return features_mfcc_delta1_delta2

def wavtomfcc(wavdict,fs):
	features_mfcc = np.zeros((len(wavdict),59,26))
	delta1 = np.zeros((len(wavdict),59,26))
	delta2 = np.zeros((len(wavdict),59,26))
	for i in range (len(wavdict)):
		features_mfcc[i] = mfcc(wavdict[i],samplerate=fs,numcep=26)#####mfcc
		delta1[i]=delta(features_mfcc[i],1)#####mfcc
		delta2[i]=delta(delta1[i],1) #####mfcc 
	return features_mfcc,delta1,delta2
    
def weight_variable(shape,name):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape,name):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial,name=name)
########################## 1D-CNN ###########################################
def conv1d(x,W):
    return tf.nn.conv1d(x,W,2,padding='SAME') #第三个参数是向下卷积的个数/每次
    
def max_pool(x):
    return tf.nn.pool(x,window_shape=[20],pooling_type="MAX",strides=[3],padding="SAME")#window_shape参数是进行池化时一数列的大小，strides是向下移动的个数
#20,4
with tf.name_scope('input_1D'):
    x_1D = tf.placeholder(tf.float32,(None,4800),name='x-input_1D')
    with tf.name_scope('x_image_1D'):
        x_image_1D = tf.reshape(x_1D,[-1,4800,1],name='x_image_1D')

with tf.name_scope('Conv1_1D'):
    with tf.name_scope('W_conv1_1D'):
        W_conv1_1D = weight_variable([300,1,10],name='W_conv1_1D')
    with tf.name_scope('b_conv1_1D'):  
        b_conv1_1D = bias_variable([10],name='b_conv1_1D')
    with tf.name_scope('conv1d_1_1D'):
        conv1d_1_1D = conv1d(x_image_1D,W_conv1_1D) + b_conv1_1D
    print(np.shape(conv1d_1_1D))
    with tf.name_scope('relu_1D'):
        h_conv1_1D = tf.nn.relu(conv1d_1_1D)
    with tf.name_scope('h_pool1_1D'):
        h_pool1_1D = max_pool(h_conv1_1D)
    print(np.shape(h_pool1_1D))
    
with tf.name_scope('Conv2_1D'):
    with tf.name_scope('W_conv2_1D'):
        W_conv2_1D = weight_variable([270,10,20],name='W_conv2_1D')
    with tf.name_scope('b_conv2_1D'):  
        b_conv2_1D = bias_variable([20],name='b_conv1_1D')
    with tf.name_scope('conv1d_2_1D'):
        conv1d_2_1D = conv1d(h_pool1_1D,W_conv2_1D) + b_conv2_1D
    print(np.shape(conv1d_2_1D))
    with tf.name_scope('relu_1D'):
        h_conv2_1D = tf.nn.relu(conv1d_2_1D)
    with tf.name_scope('h_pool2_1D'):
        h_pool2_1D = max_pool(h_conv2_1D)
    print(np.shape(h_pool2_1D))

with tf.name_scope('Conv3_1D'):
    with tf.name_scope('W_conv3_1D'):
        W_conv3_1D = weight_variable([240,20,30],name='W_conv3_1D')
    with tf.name_scope('b_conv3_1D'):  
        b_conv3_1D = bias_variable([30],name='b_conv3_1D')
    with tf.name_scope('conv1d_3_1D'):
        conv1d_3_1D = conv1d(h_pool2_1D,W_conv3_1D) + b_conv3_1D
    print(np.shape(conv1d_3_1D))
    with tf.name_scope('relu_1D'):
        h_conv3_1D = tf.nn.relu(conv1d_3_1D)
    with tf.name_scope('h_pool3_1D'):
        h_pool3_1D = max_pool(h_conv3_1D)
    print(np.shape(h_pool3_1D))

with tf.name_scope('Conv4_1D'):
    with tf.name_scope('W_conv4_1D'):
        W_conv4_1D = weight_variable([210,30,20],name='W_conv4_1D')
    with tf.name_scope('b_conv4_1D'):  
        b_conv4_1D = bias_variable([20],name='b_conv4_1D')
    with tf.name_scope('conv1d_4_1D'):
        conv1d_4_1D = conv1d(h_pool3_1D,W_conv4_1D) + b_conv4_1D
    print(np.shape(conv1d_3_1D))
    with tf.name_scope('relu_1D'):
        h_conv4_1D = tf.nn.relu(conv1d_4_1D)
    with tf.name_scope('h_pool3_1D'):
        h_pool4_1D = max_pool(h_conv4_1D)
    print(np.shape(h_pool4_1D))

with tf.name_scope('fc1_1D'):
    with tf.name_scope('h_pool1_flat_1D'):
        h_pool1_flat_1D = tf.reshape(h_pool4_1D,[-1,4*20],name='h_pool1_flat_1D')
    with tf.name_scope('keep_prob'):
        keep_prob = tf.placeholder(tf.float32,name='keep_prob') 
    with tf.name_scope('h_fc1_drop_1D'):
        h_fc1_drop_1D = tf.nn.dropout(h_pool1_flat_1D,keep_prob,name='h_fc1_drop_1D')
########################## 1D-CNN end #########################################
########################### MFCC ###########################################    
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32,(None,59,26),name='x-input')
    y = tf.placeholder(tf.float32,(None,2),name='y-input')
    with tf.name_scope('x_image'):
        x_image = tf.reshape(x,[-1,59,26,3],name='x_image')

with tf.name_scope('Conv1'):
    with tf.name_scope('W_conv1'):
        W_conv1 = weight_variable([3,3,3,16],name='W_conv1')
    with tf.name_scope('b_conv1'):  
        b_conv1 = bias_variable([16],name='b_conv1')
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image,W_conv1) + b_conv1
    print(np.shape(conv2d_1))    
    with tf.name_scope('relu'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('h_pool1'):
        h_pool1 = max_pool_2x2(h_conv1)
    print(np.shape(h_conv1))

with tf.name_scope('Conv2'):
    with tf.name_scope('W_conv2'):
        W_conv2 = weight_variable([3,3,16,16],name='W_conv2')
    with tf.name_scope('b_conv2'):  
        b_conv2 = bias_variable([16],name='b_conv2')
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1,W_conv2) + b_conv2
    print(np.shape(conv2d_2))
    with tf.name_scope('relu'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('h_pool2'):
        h_pool2 = max_pool_2x2(h_conv2)
    print(np.shape(h_pool2))

with tf.name_scope('Conv3'):
    with tf.name_scope('W_conv3'):
        W_conv3 = weight_variable([3,3,16,32],name='W_conv3')
    with tf.name_scope('b_conv3'):  
        b_conv3 = bias_variable([32],name='b_conv3')
    with tf.name_scope('conv2d_3'):
        conv2d_3 = conv2d(h_pool2,W_conv3) + b_conv3
    print(np.shape(conv2d_3))
    with tf.name_scope('relu'):
        h_conv3 = tf.nn.relu(conv2d_3)
    with tf.name_scope('h_pool3'):
       h_pool3 = max_pool_2x2(h_conv3)
    print(np.shape(h_pool3))        
      
with tf.name_scope('Conv4'):
    with tf.name_scope('W_conv4'):
        W_conv4 = weight_variable([3,3,32,32],name='W_conv4')
    with tf.name_scope('b_conv4'):  
        b_conv4 = bias_variable([32],name='b_conv4')
    with tf.name_scope('conv2d_4'):
       conv2d_4 = conv2d(h_pool3,W_conv4) + b_conv4
    print(np.shape(conv2d_4))  
    with tf.name_scope('relu'):       
        h_conv4 = tf.nn.relu(conv2d_4)
    with tf.name_scope('h_pool4'):
       h_pool4 = max_pool_2x2(h_conv4)
    print(np.shape(h_pool4)) 

with tf.name_scope('fc1'):
    with tf.name_scope('h_pool1_flat'):
        h_pool1_flat = tf.reshape(h_pool4,[-1,4*2*32],name='h_pool1_flat')
    with tf.name_scope('h_fc1_drop'):
        h_fc1_drop = tf.nn.dropout(h_pool1_flat,keep_prob,name='h_fc1_drop')
########################## MFCC ###########################################  
#组合特征        
with tf.name_scope('fc1'):
    with tf.name_scope('flat_1d_2d'):
        #flat_1d_2d = tf.concat([h_fc1_drop_1D,h_fc1_drop],axis=1,name='flat_1d_2d')
        h_1d_2d_flat = tf.concat([h_fc1_drop_1D,h_fc1_drop],axis=1,name='h_1d_2d_flat')
    #with tf.name_scope('h_1d_2d_flat'):
        #h_1d_2d_flat = tf.reshape(flat_1d_2d,[-1,8*2*32+2*20],name='h_1d_2d_flat')
        
with tf.name_scope('fc2'):
    with tf.name_scope('W_fc2'):
        W_fc2 = weight_variable([4*2*32+4*20,4*2*32+4*20],name='W_fc2')
    with tf.name_scope('b_fc2'):    
        b_fc2 = bias_variable([4*2*32+4*20],name='b_fc2')
    with tf.name_scope('wx_plus_b2'):
        wx_plus_b2 = tf.matmul(h_1d_2d_flat,W_fc2) + b_fc2
        
with tf.name_scope('out'):
    with tf.name_scope('W_fc3'):
        W_fc3 = weight_variable([4*2*32+4*20,2],name='W_fc3')
    with tf.name_scope('b_fc3'):    
        b_fc3 = bias_variable([2],name='b_fc3')
    with tf.name_scope('wx_plus_b3'):
        wx_plus_b3 = tf.matmul(wx_plus_b2,W_fc3) + b_fc3
    with tf.name_scope('softmax'):
        prediction = tf.nn.softmax(wx_plus_b3,name='prediction')
#加入正则化    
with tf.variable_scope("first") as scope:
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)
    tf.contrib.layers.apply_regularization(regularizer, weights_list=[W_conv1,W_conv2,W_conv3,W_fc2,W_conv1_1D,W_conv2_1D,W_conv3_1D])
    init_op = tf.global_variables_initializer()
    loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

with tf.name_scope('train_rate'):
    with tf.name_scope('learning_rate'):
        gloabl_steps = tf.Variable(0, trainable=False)  # 计数器，用来记录运行了几轮的BATCH_SIZE，初始为0，设置为不可训练
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, gloabl_steps, LEARNING_RATE_STEP,
                                           LEARNING_RATE_DECAY, staircase=True)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction),name='cross_entropy')
    
with tf.name_scope('cross_entropy1'):  
    cross_entropy1 = cross_entropy + loss                       #加正则化
#    cross_entropy1 = cross_entropy                             #不加正则化
    tf.summary.scalar('cross_entropy1',cross_entropy1)
    
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy1)
    
with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)
        
merged = tf.summary.merge_all() 

with tf.Session() as sess:
    # 初始化
    saver = tf.train.Saver(max_to_keep=1)
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('logs/train',sess.graph)
    test_writer = tf.summary.FileWriter('logs/test',sess.graph)
    train_x, train_y, fs, t1, t2 = gen_wavlist('C:/Users/10022/Desktop/1D+2D CNN/train_data2000')
    test_x, test_y, fs, t3, t4 = gen_wavlist('C:/Users/10022/Desktop/1D+2D CNN/test_data')#t1是AS，t2是Health
    scaler = preprocessing.StandardScaler().fit(train_x)
    scaler.transform(train_x)
    scaler.transform(test_x) 
    train_features_mfcc,train_delta1,train_delta2=wavtomfcc(train_x,fs)
    test_features_mfcc,test_delta1,test_delta2=wavtomfcc(test_x,fs)
    test_mfcc_delta1_delta2=mfcc_delta1_delta2(test_features_mfcc,test_delta1,test_delta2)
    print("load successfully")
 
    length=len(train_y)
    precise0 = []
    precise1 = []
    loss0 = []
    loss1 = []
    n = 200  #迭代次数
    precise00=0
    precise11=0
    loss00=0
    loss11=0          
    test_xx=test_mfcc_delta1_delta2
    test_yy=test_y        
    for i in range(n):
        #   批打乱顺序
        shuffle_ix = np.random.permutation(np.arange(len(train_x)))
        train_x = train_x[shuffle_ix]
        train_features_mfcc = train_features_mfcc[shuffle_ix]
        train_delta1 = train_delta1[shuffle_ix]
        train_delta2 = train_delta2[shuffle_ix]
        train_y = train_y[shuffle_ix]
        train_mfcc_delta1_delta2=mfcc_delta1_delta2(train_features_mfcc,train_delta1,train_delta2)        
        train_xx=train_mfcc_delta1_delta2
        train_yy=train_y
        sess.run(train_step,feed_dict={x:train_xx,y:train_yy,x_1D:train_x,keep_prob:0.5})
        precise00=sess.run(accuracy,feed_dict={x:test_xx,y:test_yy,x_1D:test_x,keep_prob:1.0})
        precise11=sess.run(accuracy,feed_dict={x:train_xx,y:train_yy,x_1D:train_x,keep_prob:1.0})
        loss00=sess.run(cross_entropy1,feed_dict={x:test_xx,y:test_yy,x_1D:test_x,keep_prob:1.0})
        loss11=sess.run(cross_entropy1,feed_dict={x:train_xx,y:train_yy,x_1D:train_x,keep_prob:1.0})           
        precise0.append(precise00)
        precise1.append(precise11)
        loss0.append(loss00)
        loss1.append(loss11)
        print("iter= " + str(i) + "; training accuracy: " + str(precise11) + ";validation accuracy: " + str(precise00));  
        print("      " + "   training loss: " + str(loss11) + ";validation loss: " + str(loss00));  
    #计算混淆矩阵
    prediction=sess.run(prediction,feed_dict={x:test_xx,y:test_yy,x_1D:test_x,keep_prob:1.0})
    TP=0#真正例
    TN=0#真反例
    FP=0#假正例
    FN=0#假反例
    for i in range(len(prediction)):
        if(prediction[i][0]>=prediction[i][1] and test_y[i][0]>=test_y[i][1]):
            TP=TP+1
        elif(prediction[i][0]<prediction[i][1] and test_y[i][0]<test_y[i][1]):
            TN=TN+1
        elif(prediction[i][0]>=prediction[i][1] and test_y[i][0]<test_y[i][1]):
            FP=FP+1
        elif(prediction[i][0]<prediction[i][1] and test_y[i][0]>=test_y[i][1]):
            FN=FN+1
    print('TP = ' + str(TP))
    print('TN = ' + str(TN))
    print('FP = ' + str(FP))
    print('FN = ' + str(FN))
    Accuracy=(TP+TN)/(TP+TN+FP+FN)
    Precision=TP/(TP+FP)
    Sensitivity=TP/(TP+FN)
    Specificity=TN/(FP+TN)
    F1_score=2*Precision*Sensitivity/(Precision+Sensitivity)
    print("The Accuracy is:    " + str(Accuracy))
    print("The Precision is:   " + str(Precision))
    print("The Sensitivity is: " + str(Sensitivity))
    print("The Specificity is: " + str(Specificity))
    print("The F1_score is:    " + str(F1_score))
    print("TP:"+str(TP)+"; TN:"+str(TN)+"; FP:"+str(FP)+"; FN:"+str(FN))
    #保存模型
    saver.save(sess, "models/heart_segments")
    #画准确曲线图与损失曲线图
    plt.figure(1)
    plt.xlim(0, n)
    plt.ylim(0, 1.1)
    plt.plot(list(range(len(precise0))),precise0,'r-',list(range(len(precise1))),precise1,'k-')
    plt.figure(2)
    plt.xlim(0, n)
    plt.plot(list(range(len(loss0))),loss0,'r-',list(range(len(loss1))),loss1,'k-')
