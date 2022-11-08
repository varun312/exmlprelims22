from tensorflow import keras
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense

# model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2), padding="same"))

# model.add(Conv2D(32, (3, 3), padding="same"))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2), padding="same"))

# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation('relu'))
# model.add(MaxPooling2D((2, 2), padding="same"))

# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
# model.add(Dense(64))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])

# model.save('kerasmodel')
model = keras.models.load_model('kerasmodel')
model.load_weights('first_try.h5')

from PIL import Image
import numpy as np

img = Image.open('C:/Varun/Codenges/ML/exun2022/prelims/task1/newdataset/test/superstar.jpg')
img = img.resize((256, 256))
img.show()
img = np.array(img)
img = img /255.0
img = img.reshape(1, 256, 256, 3)

prediction = model.predict(img)
print(prediction)
# print(np.argmax(prediction))