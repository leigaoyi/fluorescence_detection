# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 11:51:16 2019

@author: kasy
"""

import numpy as np
from tqdm import tqdm
from skimage import transform, measure, color, exposure, io, morphology
import glob
import tensorflow as tf
from model import  class_ori, classifier_v2
import os

confirm_dir = './data/enhance/*.jpg'
result_dir = './data/confirm/'
ckpt_path = './checkpoints/cell_100.ckpt'
fig_list = glob.glob(confirm_dir)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)

x = tf.placeholder(tf.float32, [1, 64, 64, 3])
y = classifier_v2(x, name='v2')

sess = tf.Session()
saver = tf.train.Saver()
init = tf.global_variables_initializer()

sess.run(init)
saver.restore(sess, ckpt_path)

for i in fig_list:
    fig = io.imread(i)
    fig_name = os.path.basename(i)
    mean = np.mean(fig)
    std = np.std(fig)
    fig_pro = (fig-mean)/std
    fig_pro = np.reshape(fig_pro, [1, 64, 64, 3])
    feed_dict = {x:fig_pro}
    
    fig_label = sess.run(y, feed_dict=feed_dict)
    if np.max(fig_label) == fig_label[0][2]:
        io.imsave(result_dir+fig_name, fig)
    
 