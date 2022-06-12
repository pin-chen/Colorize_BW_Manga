from loader import process
from models import unet_vgg16
from keras.applications.vgg16 import VGG16, preprocess_input
from skimage.transform import resize
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    classifier = VGG16(weights='imagenet', include_top=True)
    trainX1, trainY, testX1, testY = process(256, 240, 16, save=False)
    trainX2 = np.asarray([resize(gray2rgb(X[:, :, 0]), (224, 224)) for X in trainX1])
    testX2 = np.asarray([resize(gray2rgb(X[:, :, 0]), (224, 224)) for X in testX1])
    trainX = (trainX1, classifier.predict(preprocess_input(trainX2)))
    testX = (testX1, classifier.predict(preprocess_input(testX2)))
    m = unet_vgg16()
    history = m.fit(x=trainX, y=trainY, batch_size=16, epochs=200, validation_data=(testX, testY))
    m.save('Table/Unet-Classifier.h5')

    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Table/accuracy.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Table/full-loss.png')
    plt.clf()

    plt.xlim(30, 200)
    plt.plot(range(30, 200), history.history['loss'][30:])
    plt.plot(range(30, 200), history.history['val_loss'][30:])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Table/loss.png')
    plt.clf()
