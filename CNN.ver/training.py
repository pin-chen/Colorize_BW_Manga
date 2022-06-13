from loader import process
from models import embed_vgg16
from models import alpha_v2
from models import beta_v2
from keras.applications.vgg16 import VGG16, preprocess_input
from skimage.transform import resize
from skimage.color import gray2rgb
import numpy as np
import optparse
import sys

def readCommand(argv):
    parser = optparse.OptionParser(
        description='Train agent with different models, default use embed_vgg16 model')
    parser.set_defaults(alpha=False, beta=False, vgg=False)
    parser.add_option('--alpha',
                      dest='alpha',
                      action='store_true',
                      help='alpha_v2 model')
    parser.add_option('--beta',
                      dest='beta',
                      action='store_true',
                      help='beta_v2 model')
    parser.add_option('--embed_vgg16',
                      dest='vgg',
                      action='store_true',
                      help='embed_vgg16 model')
    (options, args) = parser.parse_args(argv)
    return options

if __name__ == '__main__':
    options = readCommand(sys.argv)
    classifier = VGG16(weights='imagenet', include_top=True)
    trainX1, trainY, testX1, testY = process(256, 240, 16)
    trainX2 = np.asarray([resize(gray2rgb(X[:, :, 0]), (224, 224)) for X in trainX1])
    testX2 = np.asarray([resize(gray2rgb(X[:, :, 0]), (224, 224)) for X in testX1])
    trainX = (trainX1, classifier.predict(preprocess_input(trainX2)))
    testX = (testX1, classifier.predict(preprocess_input(testX2)))
    if options.alpha:
        m = alpha_v2()
    elif options.beta:
        m = beta_v2()
    else:
        m = embed_vgg16()
    m.fit(x=trainX, y=trainY, batch_size=16, epochs=200, validation_data=(testX, testY))
    m.save('Table/classifier-ver.h5')
