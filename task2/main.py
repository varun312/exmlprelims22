from PIL import Image
import numpy as np
import os

# for i in range(1, 10):
#     im = Image.open('C:/Varun/Codenges/ML/exun2022/prelims/task2/binary/one/'+str(i)+'.jpg')
#     print(im.size)

#     left = 1300
#     top = 1300
#     right = 1500
#     bottom = 1500
    
#     im1 = im.crop((left, top, right, bottom))
#     im1 = im1.resize((2, 2))
#     im1.save('C:/Varun/Codenges/ML/exun2022/prelims/task2/newdataset/one/'+str(i)+'.jpg')
# for i in range(1, 10):
#     im = Image.open('C:/Varun/Codenges/ML/exun2022/prelims/task2/binary/zero/'+str(i)+'.jpg')
#     print(im.size)

#     left = 1300
#     top = 1300
#     right = 1500
#     bottom = 1500
    
#     im1 = im.crop((left, top, right, bottom))
#     im1 = im1.resize((2, 2))
#     im1.save('C:/Varun/Codenges/ML/exun2022/prelims/task2/newdataset/zero/'+str(i)+'.jpg')

import tensorflow as tf
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(2, 2, 3)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Activation('sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
from keras.preprocessing.image import ImageDataGenerator

batch_size = 2

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        )

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory('C:/Varun/Codenges/ML/exun2022/prelims/task2/newdataset/',
                                                 target_size = (2, 2),
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

# validation_generator = test_datagen.flow_from_directory('C:/Varun/Codenges/ML/exun2022/prelims/task1/newdataset/test',
#                                             target_size = (256, 256),
#                                             batch_size = 32,
#                                             class_mode = 'binary')

model.fit(train_generator,
        # steps_per_epoch = 160 // batch_size,
        epochs = 50,
        # validation_steps= 20 // batch_size
        )

model.save_weights('pls.h5')
