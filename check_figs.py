# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 18:53:49 2018

@author: kasy
"""

import imageio
import numpy as np
from tqdm import tqdm
from skimage import transform, measure, color, exposure
import glob
import tensorflow as tf
from model import classifier, class_ori, classifier_v2
import os


dir_path = './data/trainset_1/'
tmp_path = './data/tmp/'
check_path = './data/check/'
ckpt_path = './checkpoints/cell_200.ckpt'
anotate_path = './result/anotate_figs/'
process_dir = './data/process/'
#label_count = 'aq_22'

if not os.path.exists(anotate_path):
    os.makedirs(anotate_path)
if not os.path.exists(check_path):
    os.makedirs(check_path)
if not os.path.exists(tmp_path):
    os.makedirs(tmp_path)
if not os.path.exists(process_dir):
    os.makedirs(process_dir)    
    
    
x_node = tf.placeholder(tf.float32, [1, 64, 64, 3])
y_predict = classifier_v2(x_node)    

sess_build = tf.Session()
sess_build.run(tf.global_variables_initializer())
#=============sess saver restor==========
saver = tf.train.Saver()
saver.restore(sess_build, ckpt_path)   
    
tmp_figs_path = glob.glob(tmp_path+'/*.jpg')

for i in tqdm(range(len(tmp_figs_path))):
    base_name = os.path.basename(tmp_figs_path[i])
    fig_i = imageio.imread(tmp_figs_path[i])
    #preprocess
    mean_i = np.mean(fig_i)
    std_i = np.std(fig_i)
    fig_i_norm = (fig_i - mean_i)/std_i
    #run network
    fig_i_norm = np.reshape(fig_i_norm, [1, 64, 64, 3])
    feed_dict = {x_node:fig_i_norm}    
    
    predict_i = sess_build.run(y_predict, feed_dict=feed_dict)
    label_i = int(np.argmax(predict_i, axis=1))    
    if label_i > 0 :
        imageio.imsave(check_path+base_name, fig_i)
