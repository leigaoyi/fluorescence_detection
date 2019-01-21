# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:05:30 2018

@author: kasy
"""
from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage import io, transform
from skimage.util import random_noise
import numpy as np

# elastic_transform

#def gauss_process(img, sigma=0):
#        is_colour = len(img.shape)==3
#        sigma_ = np.random.random()*sigma
#        img = random_noise(img)
#        img = transform.rotate(img, 90*np.random.randint(0, 4))
#        return rescale_intensity(gaussian(img, sigma=sigma_, multichannel=is_colour))
    
def noise_add(img):
    img = transform.rotate(img, 90*np.random.randint(0, 4))
    #img = random_noise(img, mode='poisson')
    return img    
    
def zoom(img):
    zoom_size = np.random.randint(60, img.shape[0])
    row_index = np.random.randint(0, img.shape[0]-zoom_size)
    col_index = np.random.randint(0, img.shape[0]-zoom_size)
    crop_fig = img[row_index:(row_index+zoom_size), col_index:(col_index+zoom_size), :]
    resize_fig = transform.resize(crop_fig, [64, 64, 3], mode='reflect')
    return resize_fig 