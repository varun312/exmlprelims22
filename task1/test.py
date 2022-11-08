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

def predict(path):
    img = Image.open(path)
    img = img.resize((256, 256))
    img = np.array(img)
    img = img /255.0
    img = img.reshape(1, 256, 256, 3)

    prediction = model.predict(img)
    if float(prediction[0][0])>0.7:
        print("nike")
    else:
        print("adidas")

predict('newdataset/test/superstar.jpg')
