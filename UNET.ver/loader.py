from skimage.transform import resize
from skimage.color import rgba2rgb, gray2rgb, rgb2lab
from skimage.io import imread, imsave
import numpy as np
import pickle
import glob
import os


def process(pic_num, train_num, test_num, save=False):
    if not os.path.exists('../Pictures'):
        os.mkdir('../Pictures')
        if not os.path.exists('../Pictures/oringinal'):
            os.mkdir('../Pictures/oringinal')
    if not os.path.exists('Table'):
        os.mkdir('Table')
    if not os.path.exists('Train'):
        os.mkdir('Train')
    if not os.path.exists('Test'):
        os.mkdir('Test')

    full_paths = glob.glob('../Pictures/oringinal/*.png')
    if pic_num > len(full_paths):
        raise ValueError("Pictures are not enough.")
    if pic_num < train_num + test_num:
        raise ValueError("Wrong splitting.")

    data_path = 'Table/data.pickle'
    if not os.path.exists(data_path):
        save = True
        print("Loading image...")
        load_image(full_paths, data_path, pic_num)
    return split_data(train_num, test_num, save)


def load_image(paths, target, num):
    subpaths = np.random.choice(paths, min(num, len(paths)), replace=False)
    data = []
    i = 1
    for path in subpaths:
        if i % 10 == 0:
            print(f"Processing image ({i}/{len(subpaths)})")
        img = imread(path)
        if len(img.shape) == 2 or img.shape[2] == 1:
            img = gray2rgb(img)
        if img.shape[2] == 4:
            img = rgba2rgb(img)
        img_crop = resize(img, (256, 256))
        data.append(img_crop)
        i += 1
    print("Saving data...")
    data = np.asarray(data, dtype=float)
    with open(target, 'wb') as f:
        pickle.dump(data, f)


def split_data(train_num, test_num, save):
    with open('Table/data.pickle', 'rb') as f:

        data = pickle.load(f)
        train = data[:train_num]
        test = data[train_num:train_num + test_num]
        for i in range(len(train)):
            if save:
                imsave(f'Train/{i+1}.png', train[i])
            train[i] = rgb2lab(train[i])
        for i in range(len(test)):
            if save:
                imsave(f'Test/{i+1}.png', test[i])
            test[i] = rgb2lab(test[i])

        trainX = train[:, :, :, 0]
        trainY = train[:, :, :, 1:] / 128
        trainX = trainX.reshape(train_num, 256, 256, 1)
        trainY = trainY.reshape(train_num, 256, 256, 2)
        testX = test[:, :, :, 0]
        testY = test[:, :, :, 1:] / 128
        testX = testX.reshape(test_num, 256, 256, 1)
        testY = testY.reshape(test_num, 256, 256, 2)
        return trainX, trainY, testX, testY
