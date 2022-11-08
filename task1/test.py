from tensorflow import keras
model = keras.models.load_model('kerasmodel')
model.load_weights('first_try.h5')

from PIL import Image
import numpy as np

def predict(path):
    img = Image.open(path)
    img = img.resize((2800, 2800))
    left = 1300
    top = 1300
    right = 1500
    bottom = 1500
    img = img.crop((left, top, right, bottom))
    img = img.resize((2, 2))
    
    img = np.array(img)
    img = img /255.0
    img = img.reshape(1, 256, 256, 3)

    prediction = model.predict(img)
    if float(prediction[0][0])>0.7:
        print("nike")
    else:
        print("adidas")

predict('newdataset/test/superstar.jpg')
