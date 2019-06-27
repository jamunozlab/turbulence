########################### GeneratorTEST ###################################

import tensorflow as tf
#import glob
import matplotlib.pyplot as plt
import numpy as np
import os
#import PIL
from tensorflow.keras import layers
import time

import zipfile

from IPython import display

### GENERATOR ### image creator, creating the model #
# generating images from 7x7 to 28x28(depends on image pixel size), opposite of pooling
def make_generator_model():
    model = tf.keras.Sequential()
	 model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
#	 model.add(layers.BatchNormalization())
	 model.add(layers.LeakyReLU())

	 model.add(layers.Reshape((7,7,256)))
	 assert model.output_shape == (None, 7,7,256)

	 model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
	 assert model.output_shape == (None, 7,7,128)
#	 model.add(layers.BatchNormalization())
	 model.add(layers.LeakyReLU())

	 model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
	 assert model.output_shape == (None, 14,14,64)
#	 model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28,28,1)

    return model

# GENERATOR (untrained) creating image #
generator = make_generator_model()

noise = tf.random.normal([1,100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0,:,:,0], cmap='gray')
