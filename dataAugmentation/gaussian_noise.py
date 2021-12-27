from skimage.io import imread
import tensorflow as tf
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt

img = imread ('img.jpg')

print(img.shape)
shape = img.shape

x = tf.placeholder(dtype = tf.float32, shape = shape)# Adding Gaussian noise
noise = tf.random_normal(shape=tf.shape(x), mean=0.0, stddev=1.0,
dtype=tf.float32)
output = tf.add(x, noise)

arr_ = np.squeeze(output) # you can give axis attribute if you wanna squeeze in specific dimension
plt.imshow(arr_)
plt.show()

#pyplot.imshow(output..astype('uint8'))
# show the figure.
#pyplot.show()
