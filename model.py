# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 20:49:35 2018

@author: kasy
"""

import tensorflow as tf

slim = tf.contrib.slim

def double_block(in_x, out_ch):
    conv1 = slim.conv2d(in_x, out_ch, 3)
    conv2 = slim.conv2d(conv1, out_ch, 3)
    return conv2

def u_net(patches, reuse=False):
    '''
    patches : bs*512*512*3
    out_label : bs*512*512*1
    '''
    with tf.variable_scope('u_net', reuse=reuse):
        block1 = double_block(patches, 32)#512
        down1 = slim.max_pool2d(block1, 3, padding='SAME')
        
        block2 = double_block(down1, 64)#256
        down2 = slim.max_pool2d(block2, 3, stride=2, padding='SAME')
        
        block3 = double_block(down2, 128)
        down3 = slim.max_pool2d(block3, 4, stride=2, padding='SAME')
        
        block4 = double_block(down3, 256)#64
        
        up1 = slim.conv2d_transpose(block4, 128, 4, stride=2)#128
        concat1 = tf.concat([up1, block3], axis=3)
        block5 = double_block(concat1, 128)
        
        up2 = slim.conv2d_transpose(block5, 64, 4, stride=2)#256
        concat2 = tf.concat([up2, block2], axis=3)
        block6 = double_block(concat2, 64)
        
        up3 = slim.conv2d_transpose(block6, 32, 4, stride=2)
        concat3 = tf.concat([up3, block1], axis=3)
        block7 = double_block(concat3, 32)
        
        predict_label = slim.conv2d(block7, 1, 1, activation_fn=tf.nn.sigmoid)
        
        return predict_label
 
def classifier(x_in, reuse=False):
    with tf.variable_scope('clssify_L', reuse=reuse):
        conv1 = slim.conv2d(x_in, 32, 4)
        conv2 = slim.conv2d(conv1, 64, 4, stride=2)
        conv3 = slim.conv2d(conv2, 128, 4, stride=2)
        reshape = tf.reshape(conv3, [conv3.shape[0].value, -1])
        
        fn = slim.fully_connected(reshape, 3, activation_fn=None)
        return fn

def class_ori(x_in, reuse=False):
    with tf.variable_scope('clssify_ori', reuse=reuse):
        conv1 = slim.conv2d(x_in, 32, 4)
        conv2 = slim.conv2d(conv1, 64, 4, stride=2)
        conv3 = slim.conv2d(conv2, 128, 4, stride=2)
        reshape = tf.reshape(conv3, [conv3.shape[0].value, -1])
        
        fn = slim.fully_connected(reshape, 3, activation_fn=None)
        return fn        
    
if __name__ == '__main__':    
    x = tf.placeholder(tf.float32, [1, 512, 512, 3])        
    y = u_net(x)
    print(y.shape)
        