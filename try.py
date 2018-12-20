# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 14:06:57 2018

@author: ky
"""

import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure,draw 

fig_path_1 = './original.png'
fig_path_2 = './predict.png'

fig_1 = imageio.imread(fig_path_1)
fig_2 = imageio.imread(fig_path_2)

sample_path = './sample_1.jpg'
sample_1 = imageio.imread(sample_path)

h, w, c = sample_1.shape

crop = sample_1[2000:3000, 2000:3000, :]
imageio.imsave('crop.jpg', crop)    

pixel_crop = np.zeros_like(crop[:,:,0])
for i in range(1000):
    for j in range(1000):
        pixel = crop[i, j, :]
        pixel_crop[i, j] = (int(pixel[1])-int(pixel[2]))>20
        
pixel_crop *= 1     
show_fig = np.zeros_like(crop)
for i in range(3):
    show_fig[..., i] = crop[:,:,i]*pixel_crop
imageio.imsave('predict.jpg', 255*pixel_crop)
imageio.imsave('present.jpg', show_fig)

#===============draw counters===========
contours = measure.find_contours(pixel_crop, 0.5)

fig, (ax0,ax1) = plt.subplots(1,2,figsize=(15,15))
ax0.imshow(pixel_crop,plt.cm.gray)
ax1.imshow(pixel_crop,plt.cm.gray)
for n, contour in enumerate(contours):
    ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
ax1.axis('image')
ax1.set_xticks([])
ax1.set_yticks([])
plt.savefig('./{}.jpg'.format(str(n)))  
plt.show()
  

#crop_1 = crop[435:450, 540:550, :]
#imageio.imsave('crop_1.jpg', crop_1)

#list_pixel = []
#for i in range(crop_1.shape[0]):
#    for j in range(crop_1.shape[1]):
#        list_pixel.append(crop_1[i, j, :])
#        print(crop_1[i, j, :])
#list_pixel = np.asarray(list_pixel, dtype=np.int)        
#np.savetxt('list_crop.txt', list_pixel)    

    