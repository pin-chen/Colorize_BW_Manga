from loader import process
from models import embed_vgg16
from keras.applications.vgg16 import VGG16, preprocess_input
from skimage.transform import resize
from skimage.color import gray2rgb
import numpy as np


if __name__ == '__main__':
    classifier = VGG16(weights='imagenet', include_top=True)
    trainX1, trainY, testX1, testY = process(256, 240, 16)
    trainX2 = np.asarray([resize(gray2rgb(X[:, :, 0]), (224, 224)) for X in trainX1])
    testX2 = np.asarray([resize(gray2rgb(X[:, :, 0]), (224, 224)) for X in testX1])
    trainX = (trainX1, classifier.predict(preprocess_input(trainX2)))
    testX = (testX1, classifier.predict(preprocess_input(testX2)))
    m = embed_vgg16()
    m.fit(x=trainX, y=trainY, batch_size=16, epochs=200, validation_data=(testX, testY))
    m.save('Table/classifier-ver.h5')
