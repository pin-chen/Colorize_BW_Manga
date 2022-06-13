from loader import process
from models import unet_vgg16
from models import best_version
from keras.applications.vgg16 import VGG16, preprocess_input
from skimage.transform import resize
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
import numpy as np
import optparse
import sys

def readCommand(argv):
    parser = optparse.OptionParser(
        description='Train agent with different models, default use unet_vgg16 model')
    parser.set_defaults(alpha=False, beta=False, vgg=False)
    parser.add_option('--unet',
                      dest='unet',
                      action='store_true',
                      help='unet_vgg16 model')
    parser.add_option('--best',
                      dest='best',
                      action='store_true',
                      help='best version')
    (options, args) = parser.parse_args(argv)
    return options

if __name__ == '__main__':
    options = readCommand(sys.argv)
    if options.best:
        m = best_version()
    else:
        m = unet_vgg16()
    classifier = VGG16(weights='imagenet', include_top=True)
    trainX1, trainY, testX1, testY = process(256, 240, 16, save=False)
    trainX2 = np.asarray([resize(gray2rgb(X[:, :, 0]), (224, 224)) for X in trainX1])
    testX2 = np.asarray([resize(gray2rgb(X[:, :, 0]), (224, 224)) for X in testX1])
    trainX = (trainX1, classifier.predict(preprocess_input(trainX2)))
    testX = (testX1, classifier.predict(preprocess_input(testX2)))
    if options.best:
        m = best_version()
    else:
        m = unet_vgg16()
    history = m.fit(x=trainX, y=trainY, batch_size=16, epochs=200, validation_data=(testX, testY))
    m.save('Table/Unet-Classifier.h5')

    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Chart/accuracy.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Chart/full-loss.png')
    plt.clf()

    plt.xlim(30, 200)
    plt.plot(range(30, 200), history.history['loss'][30:])
    plt.plot(range(30, 200), history.history['val_loss'][30:])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('Chart/loss.png')
    plt.clf()
