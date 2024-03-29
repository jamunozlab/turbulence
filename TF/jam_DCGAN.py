####################### DCGAN_2.py ####################################

import tensorflow as tf
tf.enable_eager_execution()

import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import time

#from keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, Conv2DTranspose, Conv2D
from tensorflow.keras.models import Sequential, Model
#from tensorflow.keras.optimizers import Adam

##################### Create Models #######################
def generator_model():
    model = Sequential()

    #WIP
    model.add(Conv2D(1,(5,5),strides=(2,2),padding='same',use_bias=False,input_shape=(1,512,512),activation='tanh'))
	 #model = Sequential()
    #model.add(Dense(128*128*256, use_bias=False, input_shape=(512*512,)))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU())

    #model.add(Reshape((128,128,256)))

    assert model.output_shape == (None,128,128,256) # note: none is batch size
	 
    model.add(Conv2DTranspose(128,(5,5),strides=(1,1), padding='same',use_bias=False))
    assert model.output_shape == (None,128,128,128)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False))
    assert model.output_shape == (None,256,256,64)
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(1,(5,5),strides=(2,2),padding='same',use_bias=False,activation='tanh')) # try Adam for activation or sigmoid for binary
    assert model.output_shape == (None, 512,512,1)

    model.summary()

    return model

################################################

def discriminator_model():
    model = Sequential()
    model.add(Conv2D(64,(5,5),strides=(2,2),padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128,(5,5),strides=(2,2),padding='same'))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model

################################################

generator = generator_model()
discriminator = discriminator_model()

#### Define Loss Functions and the Optimizer ###

#                  Generator Loss              #

def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output),generated_output)

#                  Discriminator Loss          #

def discriminator_loss(real_output, generated_output):
    #[1,1,...,1] real output 
    real_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(real_output),logits=real_output)
    #[0,0,...,0] generated images are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(generated_output),logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss

###############################################

generator_optimizer = tf.train.AdamOptimizer(1e-4)
discriminator_optimizer = tf.train.AdamOptimizer(1e-4)

################################################

### Checkpoints (object-based saving) ###

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir,"ckpt")
checkpoint =tf.train.Checkpoint(genrator_optimizer = generator_optimizer, discriminator_optimizer = discriminator_optimizer, generator = generator, discriminator = discriminator)
###                                  ###

#in_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()###############################################
################################################
### Load and Process Data ###
#(train_images,train_labels),(_,_) = tf.keras.datasets.mnist.load_data()
#print(train_images.shape)

#ufos = np.load('/home/jamunoz/shared/ufo.npz')
#train_images = ufos['x_train']

#print(train_images.shape)
#print('here')


#train_images = train_images.reshape(train_images.shape[0],28,28,1).astype('float32')
#train_images = (train_images - 127.5) / 127.5 # normalize images to [-1,1]
#print(train_images.shape)

ufos = np.load('/home/jamunoz/shared/ufo.npz')
train_images = ufos['x_train']
#print(train_images.shape)
#train_images = train_images.reshape(train_images.shape[0],28,28,1).astype('float32')



BUFFER_SIZE = 40
BATCH_SIZE = 40



# create batches and shuffle dataset
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

##########################################################
################### Set up GANs for Training #############
# define training parameters

EPOCHS = 1
#noise_dim = 100
noise_dim = 512*512
num_examples_to_generate = 16

# We'll re-use this random vector used to seed generator to improve
random_vector_for_generation = tf.random_normal([num_examples_to_generate,noise_dim])
# define training method
def train_step(images, degraded_images=None):
   # generating noise from a normal distribution
    if degraded_images is None:
        noise = tf.random_normal([BATCH_SIZE, noise_dim])
    else:
        noise = degraded_images

    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
      generated_images = generator(noise, training=True)
      real_output = discriminator(images, training=True)
      generated_output = discriminator(generated_images, training=True)
      gen_loss = generator_loss(generated_output)
      dis_loss = discriminator_loss(real_output,generated_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
    gradients_of_discriminator =dis_tape.gradient(dis_loss,discriminator.variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))

##############

train_step = tf.contrib.eager.defun(train_step)
degraded_images = ufos['y_train']
degraded_images = degraded_images.reshape(40, 512*512)

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for images in dataset:
      train_step(images, degraded_images=degraded_images)

#    display.clear_output(wait=True)
    generate_and_save_images(generator, epoch + 1, random_vector_for_generation)

# saving (checkpoint) the model every 10 epochs
  if (epoch + 1) % 10 == 0:
	  checkpoint.save(file_prefix = checkpoint_prefix)

  print ('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

# generating after the final epoch
  #display.clear_output(wait=True)
  generate_and_save_images(generator, epochs, random_vector_for_generation)

### Generate and save images ###

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4,4))

    for i in range (predictions.shape[0]):
      plt.subplot(4,4,i+1)
      plt.imshow(predictions[i,:,:,0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

####################### Train the GANs ####################
#%% train
train(train_dataset, EPOCHS)

