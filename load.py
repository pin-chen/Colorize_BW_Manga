import numpy as np
import pickle
import glob
import cv2


def load_image(paths):
    np.random.seed(7414)
    subpaths = np.random.choice(paths, 800, replace=False)
    data = []
    i = 1
    for path in subpaths:
        if i % 100 == 0:
            print(f"Processing image ({i}/{len(subpaths)})")
        img = cv2.imread(path)
        img_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        data.append([img_gray, img])
        i += 1
    print("Saving data...")
    with open('Table/color_img_data', 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    load_image(glob.glob('Picture/*'))

