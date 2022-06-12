from keras.models import load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from skimage.transform import resize
from skimage.color import lab2rgb, rgb2lab, rgb2gray, gray2rgb, rgba2rgb
from skimage.io import imread, imsave
import numpy as np
import glob
import os


if not os.path.exists('Result'):
    os.mkdir('Result')

m = load_model('Table/Unet-Classifier.h5')
paths = glob.glob('Test/*')

classifier = VGG16(weights='imagenet', include_top=True)

for i in range(len(paths)):
    img = imread(paths[i])
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = gray2rgb(img)
    if img.shape[2] == 4:
        img = rgba2rgb(img)
    img_lab = rgb2lab(resize(img, (256, 256)))
    X1 = img_lab[:, :, 0]
    X2 = np.asarray([resize(gray2rgb(rgb2gray(img)), (224, 224))])
    X1 = X1.reshape(1, 256, 256, 1)
    X = (X1, classifier.predict(preprocess_input(X2)))
    Y_predict = m.predict(X)
    Y_predict *= 128
    result = np.zeros((256, 256, 3))
    result[:, :, 0] = X1[0][:, :, 0]
    result[:, :, 1:] = Y_predict[0]
    imsave(f'Result/{i+1}.png', lab2rgb(result))
