from skimage.transform import resize
from skimage.color import rgb2lab
from skimage.io import imread
import numpy as np
import pickle


def load_image(paths, target):
    subpaths = np.random.choice(paths, 10, replace=False)
    data = []
    for path in subpaths:
        img_lab = rgb2lab(resize(imread(path), (1080, 1920)))
        data.append(img_lab)
    print("Saving data...")
    with open(target, 'wb') as f:
        pickle.dump(data, f)
    print("Saved!!!")
