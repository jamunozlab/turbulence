########################### GANtemplate.py ##############################

import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
#import PIL
from tensorflow.keras import layers
import time

#import zipfile

#from IPython import display

# loading data from folder and labeling through folders, prepare dataset #
#local_zip = '/TF/Images/Turbulence_Images.zip'
#zip_ref = zipfile.ZipFile(local_zip, 'r')
#zip_ref.extractall('/TF/Images/Turbulence_Images')
#local_zip = '/TF/Images/Validation_Turbulence_Images.zip'
#zip_ref = zipfile.ZipFile(local_zip, 'r')
#zip_re.extractall('/TF/Images/Validation_Turbulence_Images')
#zip_ref.close()

# Directory with images #
#train_images = os.path.join('/TF/Images/Turbulence_Images')

# Directory with validation images #
#test_images = os.path.join('/TFImages/Validation_Turbulence_Images')

(train_images,train_labels), (_,_) = tf.keras.datasets.mnist.load_data()

#import cPickle, gzip, numpy

# Load the dataset
#f = gzip.open('mnist.pkl.gz', 'rb')
#train_set, valid_set, test_set = cPickle.load(f)
#f.close()

# Data prep/normalization #
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5

BUFFER_SIZE = 60000# number of images #
BATCH_SIZE = 256# depends on number of images #

# batch and shuffle data #
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

### GENERATOR ### image creator, creating the model #
# generating images from 7x7 to 28x28(depends on image pixel size), opposite of pooling
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7,7,256)))
    assert model.output_shape == (None, 7,7,256)

    model.add(layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7,7,128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14,14,64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28,28,1)

    return model

# GENERATOR (untrained) creating image #
generator = make_generator_model()

noise = tf.random.normal([1,100])
generated_image = generator(noise, training=False)

#plt.imshow(generated_image[0,:,:,0], cmap='gray')

### DISCRIMINATOR (untrained) ### judges images created by GENRATOR, CNN based image classifier #

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[28, 28, 1])) #depends on image size by generator

    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5,5), strides=(2,2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = make_discriminator_model()
decision =discriminator(generated_image)
print (decision)

# defining the Loss and Optimizer functions #
# Loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# DISCRIMINATOR Loss #
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# GENERATOR Loss #
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake.output)

# Optimizers #
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# Save Checkpoints #

#checkpoint_dir = './training_checkpoints'
#checkpoints_prefix = os.path.join(checkpoint_dir, 'ckpt')
#checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
#                                 discriminator_optimizer = discriminator_optimizer,
#											generator = generator,
#											discriminator = discriminator)

### Training Loop ###
EPOCHS = 50 
noise_dim = 100
num_examples_to_generate = 16

# will reuse seed overtime, to visualize progress in the animated GIF
seed=tf.random.normal([num_examples_to_generate, noise_dim])

##################################

def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
      generated_images = generated(noise, training = True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      dis_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = dis_tape.gradient(dis_loss, discriminator.trainable_variable)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variable))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variable))

def train(dataset, epochs):
  for epoch in range(epochs):
	  start = time.time()

	  for image_batch in dataset:
		  train_step(image_batch)

	  # Producing images for  GIFs
	  display.clear_output(wait=True)
	  generate_and_save_images(generator, epoch+1, seed)
     # save model after every 15 epochs
  if (epoch+1) % 15 == 0:
       checkpoint.save(file_prefix = checkpoint_prefix)

       print ('Time for epoch {} is {} sec'. format(epoch+1, time.time()-start))

# generate after the final epoch
###display.clear_output(wait=True)
###generate_and_save_images(generator, epochs, seed)

# Generate and Save Images  #
def generate_and_save_images(model, epoch, test_input):
  predictions =model(test_input, training=False)  # training=False is so all layers run in inference mode

###################################################
  fig = plt.figure(figsize=(4,4))

  for i in range(predictions.shape[0]):
      plt.subplot(4,4,i+1)
      plt.imshow(predictions[i,:,:,0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

### Train the Model ###

# %%time 
train(train_dataset, EPOCHS)
