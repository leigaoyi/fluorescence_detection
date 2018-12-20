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
batch_size = 5
epoch = 4000

fig_data = []
fig_ori_data = []
label_data = []

if not os.path.exists(check_dir):
    os.makedirs(check_dir)

for i in range(3):
    fig_path_list = glob.glob(fig_path+str(i)+'/*.jpg')
    for j in fig_path_list:
        fig = imageio.imread(j)
        fig_ori_data.append(fig)
        fig_L = color.rgb2grey(fig)
        fig_data.append(fig_L)
        label_data.append(i)
        
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

y_predict = classifier(x_node)
y_ori_predict = class_ori(x_ori_node)
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
for i in range(epoch):
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
        sess.run(opt_train, feed_dict=feed_dict)
        sess.run(opt_ori, feed_dict=feed_dict)
        loss_ori_val = sess.run(loss_ori, feed_dict=feed_dict)
        convert_loss_val = sess.run(loss_convert, feed_dict=feed_dict)
    print('epoch {0} ori loss {1:.4f} convert loss {2:.4f}'.format(i, loss_ori_val, convert_loss_val))
    if (i+1)%100 == 0 :
        saver.save(sess, check_dir+'cell_{0}.ckpt'.format(i+1))
