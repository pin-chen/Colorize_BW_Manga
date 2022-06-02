from keras.models import load_model
from skimage.transform import resize
from skimage.color import lab2rgb, rgb2lab, rgb2gray
from skimage.io import imread, imsave
import numpy as np
import glob
import os


if not os.path.exists('Result'):
    os.mkdir('Result')

m = load_model('Table/beta_model.h5')
paths = glob.glob('Test/*')

for i in range(len(paths)):
    img_lab = rgb2lab(resize(imread(paths[i]), (256, 256)))
    X = img_lab[:, :, 0]
    X = X.reshape(1, 256, 256, 1)
    Y_predict = m.predict(X)
    Y_predict *= 128
    result = np.zeros((256, 256, 3))
    result[:, :, 0] = X[0][:, :, 0]
    result[:, :, 1:] = Y_predict[0]
    imsave(f'Result/{i}.png', lab2rgb(result))
