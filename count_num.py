# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 14:27:40 2018

@author: kasy
"""

from skimage import measure,draw 
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pickle
from tqdm import tqdm

cell_path = './data/anotate_label.jpg'

fig_label = imageio.imread(cell_path)

contours = measure.find_contours(fig_label, 0.5)

#fig, (ax0,ax1) = plt.subplots(1,2,figsize=(20,20))
#ax0.imshow(fig_label,plt.cm.gray)
#ax1.imshow(fig_label,plt.cm.gray)
count = 0
container_point_list = []
task = 'label'
cell_area = 175
predict_area = 100
for contour in contours:
    h_min = np.min(contour, axis=0)[0]
    h_max = np.max(contour, axis=0)[0]
    
    col_list = contour[:, 1]
    w_min = np.min(col_list)
    w_max = np.max(col_list)
    # cell 180, predict 512
    if task == 'cell':
        if ((h_max-h_min)*(w_max-w_min)) > cell_area  :  
            count += 1
          #  container_point_list.append([int((h_min+h_max)/2), int((w_min+w_max)/2)])
        if ((h_max-h_min)*(w_max-w_min)) > cell_area*2 and ((h_max-h_min)*(w_max-w_min))<cell_area*5 :
            count += ((h_max-h_min)*(w_max-w_min))//cell_area-1
    else :
        if ((h_max-h_min)*(w_max-w_min)) > predict_area  :  
            count += 1    
            container_point_list.append([int((h_min+h_max)/2), int((w_min+w_max)/2)])      
print(count)    
np.savetxt('./data/point_list.txt', container_point_list)    