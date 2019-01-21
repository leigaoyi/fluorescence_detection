from skimage.filters import gaussian
from skimage.exposure import rescale_intensity
from skimage import io, transform
from skimage.util import random_noise
import numpy as np
import os


fig_path = './data/test.jpg'

fig_arr = io.imread(fig_path)

fig_noise = random_noise(fig_arr, mode='poisson')

io.imsave('./data/test_possion.jpg', fig_noise)