############################### CodeTemplate.py ########################

# getting data, importing images and unzipping #
import os
import zipfile

local_zip = '/desktop/Images/Turbulence_Images.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/desktop/Images/Turbulence_Images')
local_zip = 'desktop/Images/Validation_Turbulence_Images.zip'
zip_red = zipfile.ZipFile(local_zip, 'r')
zip_re.extractall('desktop/Images/Validation_Turbulence_Images')
zip_ref.close()

# Directory with no noise training images
train_no_noise = os.path.join('/desktop/Images/Turbulence_Images/no_noise')

# Directory with noise images
train_noise = os.path.join('/desktop/Images/Turbulence_Images/noise')

# Directory with  validation no noise images
validation_no_noise = os.path.join('/desktop/Images/validation_Turbulence_Images/validation_no_noise')

# Directory with validation noise images

validation_noise = os.path.join('/desktop/Images/validation_Turbulence_Images/validation_noise')

### Building Model ###

import tenorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

model = tf.keras.models.Sequential([
# desired size 150x150 with 3 byes of color
# first convolution, 3x3 filter, and 2x2 pooling
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
	 tf.keras.layers.MaxPooling2D(2,2),
# second convolution, 3x3 filter, and 2x2 pooling
	 tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
	 tf.keras.layers.MaxPooling2D(2,2),
# third convolution, 3x3 filter, and 2x2 pooling
	 tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
	 tf.keras.layers.MaxPooling2D(2,2),
# fourth convolution, 3x3 filter, and 2x2 pooling
	 tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
	 tf.keras.layers.MaxPooling2D(2,2),

# Flatten results for DNN
    tf.keras.layers.Flatten(),
# 512 neuron hidden layer
	 tf.keras.layers.Dense(512, activation='relu')

# 1 neuron output, sigmoid: 1 for 'no_filter', 0 for 'filter'
    tf.keras.layers.Dense(1, activation = 'sigmoid')
])

# model summary
model.summary()

# model compile #
# potential optmizers: SGD, Adam, Adagrad. Could alternate between them to see results. Will use RMSprop for automated learning rate tuning

from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy', 
              optimizer = RMSprop(1r=0.001),
				  metrics =['acc'])

# Pre-Processing Data, normalizing data #
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Image re-scale by 1./255
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale 1./255)

# training images flow in batches of 128
train_generator = train_data_gen.flow_from_directory('desktop/Turbulence_Images/', #source drectory for training images
    target_size =(150,150), #resizing images to 150x150
    batch_size = 128, #images per batch
    class_mode = 'binary') #need binary since we use binary_crossentropy

# validation images use smaller batch, less validation images used 1/4 of training images
validation_generator = validation_data_gen.flow_from_directory('/desktop/valaidation_Turbulence_Images/',
    target_size = (150,150),
	 batch_size = 32,
	 class_mode = 'binary')

# Training #

# epochs and steps per epoch subject to change depending on batch and number of images

history =model.fit_generator(
      train_generator,
		steps_per_epoch = 8,
		epochs = 15,
		verbose = 1,
		validation_data = validation_generator,
		validation_steps = 8)

# Running the Model and Predicting #
import numpy as np
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():
	#predicting
	path = '/content/' + fn
	img = image.load_img(path, target_size =(150,150))
	x = image.img_to_array(img)
	x = np.expad_dims(x, axis=0)

	images = np.vstack([x])
	classes = model.predict(images, batch_size=10)
	print (classes[0])
	if classes[0]>0.5:
		print(fn + "has noise")
	if classes[0]<=0.5:
		print(fn + "has no noise")

## Clean up to free memory resources

#import os, signal
#os.kill(os.getpid(), signal,SIGKILL)
