########################### GeneratorTEST ###################################

import tensorflow as tf
#import glob
#import matplotlib.pyplot as plt
import numpy as np
import os, sys
#import PIL
from tensorflow.keras import layers
import time

import zipfile

#from IPython import display

### GENERATOR ### image creator, creating the model #
# generating images from 7x7 to 28x28(depends on image pixel size), opposite of pooling
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
#   model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7,7,256)))
    assert model.output_shape == (None, 7,7,256)

    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7,7,128)
#   model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14,14,64)
#   model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28,28,1)

    return model

# GENERATOR (untrained) creating image #
generator = make_generator_model()

noise = tf.random.normal([1,100])
generated_image = generator(noise, training=False)

image = generated_image[0, :, :, 0]

generated_image.eval(feed_dict={image:0.0})

#np.savetxt('gen_image1', image)

#plt.imshow(generated_image[0, :, :, 0], cmap='gray')

##############################################

### Generate and save images ###
#def generate_and_save_images(model,epoch,test_input):
#    predictions =model(test_input, training=False)

#	 fig = plt.figure(figsize=(4,4))

#	 for i in range(predictions.shape[0]):
#		 plt.subplot(4,4,i+1)
#		 plt.imshow(predictions[i, :, :, 0] * 127.5, cmap='gray')
#		 plt.axis('off')

#    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
#	 plt.show()

##############################################

#print(generated_image[0,:,:, 0]) 

###print(tf.print(generated_image[0,:,:,0], output_stream=sys.stdout))

#image = (generated_image[0,:,:,0])
#np.savetxt('gen_image1', image)
#np.savetxt('gen_image1', generated_image[0,:,:,0])
#t = generator(noise, training=False)
#tf.print(t, [t])
#t = tf.print(t, [t])
#result = t + 1


#print(generated_image.np())

#plt.imshow(generated_image[0,:,:,0], cmap='gray')

### storing data ###

#np.savetxt('gen_image1', generated_image)

#with open ('generated_image','a') as f:
#    f.write('image matrix')
#	 for data in generated_image:
#		 f.write(data+'\n')
