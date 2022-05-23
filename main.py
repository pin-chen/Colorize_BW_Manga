from load import load_image
from model import build_model, apply_model

from skimage.color import lab2rgb, rgb2gray
from skimage.io import imsave
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
img_lab = data[np.random.randint(0, len(data))]
L = img_lab[:, :, 0]
ab = img_lab[:, :, 1:]/128
L = L.reshape(1, 400, 400, 1)
ab = ab.reshape(1, 400, 400, 2)

m = build_model()
output = apply_model(m, L, ab, epoch=500)

color_result = np.zeros((400, 400, 3))
color_result[:, :, 0] = L[0][:, :, 0]
color_result[:, :, 1:] = output[0]
imsave("grayscale.png", rgb2gray(lab2rgb(color_result)))
imsave("color_result.png", lab2rgb(color_result))
