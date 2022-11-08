from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), padding="same"))

model.add(Conv2D(32, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), padding="same"))

model.add(Conv2D(64, (3, 3), padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D((2, 2), padding="same"))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

batch_size = 16

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory('C:/Varun/Codenges/ML/exun2022/prelims/task1/newdataset/train',
                                                 target_size = (256, 256),
                                                 batch_size = batch_size,
                                                 class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory('C:/Varun/Codenges/ML/exun2022/prelims/task1/newdataset/test',
                                            target_size = (256, 256),
                                            batch_size = 32,
                                            class_mode = 'binary')

model.fit(train_generator,
        steps_per_epoch = 160 // batch_size,
        epochs = 50,
        validation_data = validation_generator,
        validation_steps= 20 // batch_size)

model.save_weights('first_try.h5')