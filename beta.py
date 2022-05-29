from load import load_image
from model import build_model, apply_model
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose
from keras.layers import Activation, Dense, Dropout, Flatten
from tensorflow.keras.layers import BatchNormalization
from keras.models import Sequential
from tensorflow.keras.utils import  array_to_img, img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2lab, lab2rgb, rgb2gray, xyz2lab
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
import glob

model = Sequential()
model.add(InputLayer(input_shape=(None, None, 1)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same', strides=2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', strides=2))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(2, (3, 3), activation='tanh', padding='same'))
model.compile(optimizer='rmsprop',
			loss='mse')
i = 0
for path in glob.glob('D:\github\Introduction-to-AI-Final-Project\Picture\oringinal/*'):
    i += 1
    if i > 10:
        break
    image = img_to_array(load_img(path))
    image = np.array(image, dtype=float)
    X = rgb2lab(1.0/255*image)[:,:,0]
    Y = rgb2lab(1.0/255*image)[:,:,1:]
    Y /= 128
    X = X.reshape(1, 1080, 1920, 1)
    Y = Y.reshape(1, 1080, 1920, 2)
    model.fit(x=X, y=Y,	batch_size=1, epochs=5)

print(model.evaluate(X, Y, batch_size=1))
image = img_to_array(load_img('D:\github\Introduction-to-AI-Final-Project\Picture\manga (1).jpg'))
image = np.array(image, dtype=float)
X = rgb2lab(1.0/255*image)[:,:,0]
X = X.reshape(1, 1200, 816, 1)

output = model.predict(X)
output *= 128
# Output colorizations
cur = np.zeros((1200, 816, 3))
cur[:,:,0] = X[0][:,:,0]
cur[:,:,1:] = output[0]
imsave("img_result.png", lab2rgb(cur))
imsave("img_gray_version.png", rgb2gray(lab2rgb(cur)))
