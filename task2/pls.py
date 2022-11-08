from tensorflow import keras
import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(2, 2, 3)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Activation('sigmoid')
])

model.load_weights('pls.h5')

from PIL import Image
import numpy as np

im = Image.open('C:/Varun/Codenges/ML/exun2022/prelims/task2/newdataset/one/5.jpg')
# print(im.size)

# left = 1300
# top = 1300
# right = 1500
# bottom = 1500

# im1 = im.crop((left, top, right, bottom))
# im1 = im1.resize((2, 2))
# im1.save('C:/Varun/Codenges/ML/exun2022/prelims/task2/newdataset/one/'+str(i)+'.jpg')
# img = img.resize((2, 2))
# img.show()
img = np.array(im)
img = img /255.0
img = img.reshape(1, 2, 2, 3)

prediction = model.predict(img)
print(prediction)
