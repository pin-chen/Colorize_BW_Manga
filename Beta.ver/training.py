from loader import process
from models import alpha_v2, beta_v2, test_vgg

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.models import load_model

from skimage.transform import resize
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage.io import imread, imsave

import numpy as np
import pickle
import glob
import os


if __name__ == '__main__':
    trainX, trainY, testX, testY = process(256, 240, 16)
    m = beta_v2()
    m.fit(x=trainX, y=trainY, batch_size=16, epochs=128, validation_data=(testX, testY))
    m.save('Table/beta_model.h5')
