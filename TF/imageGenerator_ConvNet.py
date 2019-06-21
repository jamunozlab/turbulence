
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu',
	                        input_shape=(300, 300, 3)),
	 tf.keras.layers.MaxPooling2D(2,2),
	 tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
	 tf.keras.layers.MaxPooling2D(2,2),
	 tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
	 tf.keras.layers.MaxPooling2D(2,2),
	 tf.keras.layers.Flatten(),
	 tf.keras.layers.Dense(512, activation='relu'),
	 tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

# train model from data #

model.fit_generator

# compile #

model.compile(loss='binary_crossentropy',
             optimizer=RMSprop(1r=0.001),
				 metrics=['acc'])

history = model.fit_generator(
      train_generator,
		steps_per_epoch=8,
		epochs=15,
		validation_data = validation_generator,
		validation_steps = 8,
		verbose = 2)

# prediction code #

import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():

   # predicting images 
  path = '/content/' + fn
  img = image.load_img(path, target_size =(300, 300))
  x = image.img_to_array(img)
  x =np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])

  if classes =[0]>0.5:
	  print(fn + " is a human")
  else:
	  print(fn + " is a horse")
