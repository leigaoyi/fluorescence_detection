# -*- coding: utf-8 -*-
"""
Created on Thu Dec 27 13:31:33 2018

@author: kasy
"""

from skimage import io
import os
import glob

tmp_path = './data/tmp/*.jpg'
check_path = './data/check/'

enhance_dir = './data/enhance/'
if not os.path.exists(enhance_dir):
    os.makedirs(enhance_dir)
    
tmp_fig_list = glob.glob(tmp_path)
for i in tmp_fig_list:
    i_name = os.path.basename(i)
    if not os.path.exists(check_path+i_name):
        fig = io.imread(i)
        io.imsave(enhance_dir+i_name, fig)
