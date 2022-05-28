from load import load_image
from model import simple_cnn

from skimage.transform import resize
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.io import imread, imsave
from keras.models import load_model
import numpy as np
import pickle
import glob
import os


data_path = 'Table/color_img_data.pickle'
if not os.path.exists(data_path):
    print("Loading image...")
    if not os.path.exists('Table'):
        os.mkdir('Table')
    load_image(glob.glob('Picture/*'), data_path)


with open(data_path, 'rb') as f:
    data = pickle.load(f)
    img_lab = np.asarray(data)  # adjust the training data size
    L = img_lab[:, :, :, 0]
    ab = img_lab[:, :, :, 1:]/128
    L = L.reshape(len(img_lab), 256, 256, 1)
    ab = ab.reshape(len(img_lab), 256, 256, 2)
    m = simple_cnn()
    m.fit(x=L, y=ab, batch_size=32, epochs=3)
    m.save('Table/simple_model.h5')

m = load_model('Table/simple_model.h5')
img = imread('test.png')
img_lab = rgb2lab(resize(img[:, :, :3], (256, 256)))
L = img_lab[:, :, 0]
L = L.reshape(1, 256, 256, 1)
output = m.predict(L)
output *= 128
color_result = np.zeros((256, 256, 3))
color_result[:, :, 0] = L[0][:, :, 0]
color_result[:, :, 1:] = output[0]
imsave("grayscale.png", rgb2gray(lab2rgb(color_result)))
imsave("color_result.png", lab2rgb(color_result))
