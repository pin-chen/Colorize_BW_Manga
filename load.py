from skimage.transform import resize
from skimage.color import rgb2lab
from skimage.io import imread
import numpy as np
import pickle


def load_image(paths, target):
    subpaths = np.random.choice(paths, 800, replace=False)
    data = []
    i = 1
    for path in subpaths:
        if i % 100 == 0:
            print(f"Processing image ({i}/{len(subpaths)})")
        img_lab = rgb2lab(resize(imread(path), (400, 400)))
        data.append(img_lab)
        i += 1
    print("Saving data...")
    with open(target, 'wb') as f:
        pickle.dump(data, f)
