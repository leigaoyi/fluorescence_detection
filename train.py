# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 09:44:08 2018

@author: kasy
"""

import tensorflow as tf
from model import classifier, class_ori
import numpy as np

from skimage import color, transform
from tqdm import tqdm
import imageio
import os
import glob
from opt import elastic_transform    

fig_path = './data/train/'
check_dir = './checkpoints/'
result_dir = './result/'
test_path = './data/test/'
batch_size = 5
epoch = 400

fig_data = []
fig_ori_data = []
label_data = []
test_fig_data = []
test_label_data = []

if not os.path.exists(check_dir):
    os.makedirs(check_dir)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)    

for i in range(3):
    fig_path_list = glob.glob(fig_path+str(i)+'/*.jpg')
    for j in fig_path_list:
        fig = imageio.imread(j)
        fig_ori_data.append(fig)
        fig_L = color.rgb2grey(fig)
        fig_data.append(fig_L)
        label_data.append(i)
        
for i in range(3):
    fig_path_list = glob.glob(test_path+str(i)+'/*.jpg')        
    for j in fig_path_list:
        test_fig = imageio.imread(j)
        test_fig_data.append(test_fig)
        test_label_data.append(i)
        
data_num = len(fig_data)        
shuffle = [i for i in range(data_num)] 
np.random.shuffle(shuffle)

def data_augment(x):
    '''
    data augmentation
    x : [batch_size, 64, 64, 1]
    output : flip, elastic
    '''
    x = elastic_transform(x, alpha=720, sigma=20, is_random=True)
    x = transform.rotate(x, 90*np.random.randint(0, 3))
    return x

batch_idx = data_num // batch_size
#===============build model==============
x_node = tf.placeholder(tf.float32, [batch_size, 64, 64, 1])
x_ori_node = tf.placeholder(tf.float32, [batch_size, 64, 64, 3])
y_node = tf.placeholder(tf.int32, [batch_size])
x_check_node = tf.placeholder(tf.float32, [1, 64, 64, 3])

y_predict = classifier(x_node)
y_ori_predict = class_ori(x_ori_node, reuse=False)

y_check = class_ori(x_check_node, reuse=True) # every epoch , check the accuracy
y_check_list = []

loss_convert = tf.losses.sparse_softmax_cross_entropy(labels=y_node, logits=y_predict)
loss_ori = tf.losses.sparse_softmax_cross_entropy(labels=y_node, logits=y_ori_predict)
loss = loss_convert+ loss_ori

var_list = tf.trainable_variables()
var_convert = [i for i in var_list if 'clssify_L' in i.name]
var_ori_list = [i for i in var_list if 'clssify_ori' in i.name]
opt_m = tf.train.AdamOptimizer(0.0001)
opt_ori = tf.train.AdamOptimizer(0.0001)
opt_train = opt_m.minimize(loss_convert, var_list=var_convert)
opt_ori = opt_ori.minimize(loss_ori, var_list=var_ori_list)

#===========init ==
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
saver = tf.train.Saver(max_to_keep=10)
#===============strat train=============
for i in tqdm(range(epoch)):
    np.random.shuffle(shuffle)
    
    for j in range(batch_idx):
        train_data_batch = []
        train_label_batch = []
        ori_data_batch = []
        for k in range(batch_size): # shuffle
            index = shuffle[batch_size*j+k]
            train_data_batch.append(data_augment(fig_data[index]))
            ori_fig = fig_ori_data[index]
            ori_fig = transform.rotate(ori_fig, 90*np.random.randint(0, 4))#data augmentation
            ori_data_batch.append(ori_fig)
            train_label_batch.append(label_data[index])  
        ori_data_batch = np.reshape(ori_data_batch, [batch_size, 64, 64, 3])
        train_data_batch = np.reshape(train_data_batch, [batch_size, 64, 64, 1])                   
        # =======standard input======
        mean_ori = np.mean(ori_data_batch)
        std_ori = np.std(ori_data_batch)
        max_gray = np.max(train_data_batch)
        min_gray = np.min(train_data_batch)

        ori_data_batch = (ori_data_batch-mean_ori)/std_ori
        train_data_batch = (train_data_batch-min_gray)/(max_gray-min_gray)
        
        #===============sess run===============
        feed_dict = {x_node:train_data_batch, y_node:train_label_batch,
                     x_ori_node: ori_data_batch}
        #============run optimization==========
        #sess.run(opt_train, feed_dict=feed_dict)
        sess.run(opt_ori, feed_dict=feed_dict)
        loss_ori_val = sess.run(loss_ori, feed_dict=feed_dict)
        convert_loss_val = sess.run(loss_convert, feed_dict=feed_dict)
    
    if (i+1)%100 == 0 :
        saver.save(sess, check_dir+'cell_{0}.ckpt'.format(i+1))
        
#=========================check train every epoch=========================
    network_result = []
    test_network_result = []
    for v in fig_ori_data:
        mean_v = np.mean(v)
        std_v = np.std(v)
        v_single = (v-mean_v)/std_v
        v_single = np.reshape(v_single, [1, 64, 64, 3])
        feed_dict = {x_check_node : v_single}
        single_test = sess.run(y_check, feed_dict=feed_dict)
        network_result.append(int(np.argmax(single_test, axis=1)))
    for v in test_fig_data:
        mean_v = np.mean(v)
        std_v = np.std(v)
        v_single = (v-mean_v)/std_v
        v_single = np.reshape(v_single, [1, 64, 64, 3])
        feed_dict = {x_check_node : v_single}
        single_test = sess.run(y_check, feed_dict=feed_dict)
        test_network_result.append(int(np.argmax(single_test, axis=1)))
    acc = 0
    test_acc = 0
    for v in range(len(network_result)):
        if network_result[v] == label_data[v]:
            acc += 1
    for v in range(len(test_network_result)):
        if test_network_result[v] == test_label_data[v] :
            test_acc += 1
    epoch_acc = acc/len(network_result)  
    epoch_test_acc = test_acc/len(test_network_result)
    y_check_list.append(epoch_acc)
    print('epoch {0} ori loss {1:.4f}, acc {2:.3f} test acc {3:.3f}'.format(i, loss_ori_val, epoch_acc, epoch_test_acc))
    if (i+1)%50 == 0:
        print('epoch ', i+1, ' acc {0:.3f} ;test acc {1:.3f}'.format(y_check_list[-1], epoch_test_acc))
        np.savetxt(result_dir+'/accuracy_train.txt', y_check_list)    
print('acc highest in {} epoch'.format(y_check_list.index(np.max(y_check_list))))
