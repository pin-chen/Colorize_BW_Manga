from load import load_image
from model import alpha_v2, beta_v2, test_vgg

from keras.models import load_model
from skimage.color import lab2rgb, rgb2gray
from skimage.io import imsave
import numpy as np
import pickle
import glob
import os

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from skimage.transform import resize
from skimage.color import rgb2lab
from skimage.io import imread


data_path = 'Table/color_img_data.pickle'
if not os.path.exists(data_path):
    print("Loading image...")
    if not os.path.exists('Table'):
        os.mkdir('Table')
    load_image(glob.glob('Picture/*'), data_path)


with open(data_path, 'rb') as f:
    data = pickle.load(f)
    train = np.asarray(data[:240])
    trainX = train[:, :, :, 0]
    trainY = train[:, :, :, 1:]/128
    trainX = trainX.reshape(len(train), 256, 256, 1)
    trainY = trainY.reshape(len(train), 256, 256, 2)
    val = np.asarray(data[240:])
    valX = val[:, :, :, 0]
    valY = val[:, :, :, 1:] / 128
    valX = valX.reshape(len(val), 256, 256, 1)
    valY = valY.reshape(len(val), 256, 256, 2)
    vgg = VGG16(weights='imagenet', include_top=True)
    train_X2 = np.asarray([resize(lab2rgb(data[i]), (224, 224)) for i in range(0, 240)])
    val_X2 = np.asarray([resize(lab2rgb(data[i]), (224, 224)) for i in range(240, 256)])
    m = test_vgg()
    m.fit(x=(trainX, vgg.predict(train_X2)), y=trainY, batch_size=16, epochs=128, validation_data=((valX, vgg.predict(val_X2)), valY))
    m.save('Table/vgg_model.h5')

vgg = VGG16(weights='imagenet', include_top=True)
m = load_model('Table/vgg_model.h5')
img_lab = rgb2lab(resize(imread('test.png'), (256, 256)))
X = img_lab[:, :, 0]
X = X.reshape(1, 256, 256, 1)
X2 = np.asarray([resize(imread('test.png'), (224, 224))])
output = m.predict((X, vgg.predict(X2)))
output *= 128
color_result = np.zeros((256, 256, 3))
color_result[:, :, 0] = X[0][:, :, 0]
color_result[:, :, 1:] = output[0]
imsave("grayscale.png", rgb2gray(lab2rgb(color_result)))
imsave("color_result.png", lab2rgb(color_result))

