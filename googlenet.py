import theano
import theano.tensor as T
import numpy as np
from sklearn_theano.feature_extraction.caffe.googlenet import create_theano_expressions
from sklearn_theano.feature_extraction.caffe.googlenet_class_labels import _class_names
from PIL import Image


def apply_google_net(x):
    mean_values = np.array([104, 117, 123]).reshape((3, 1, 1))
    # Convert RGB to BGR
    xx = x[:, ::-1, :, :] * 255.0
    xx = xx - mean_values[np.newaxis, :, :, :].astype('float32')

    net = create_theano_expressions(inputs=('data', xx))
    pre_softmax = net[0]['loss3/classifier']

    return pre_softmax.flatten(2)


def test():
    x = T.tensor4('x')
    pre_softmax = apply_google_net(x)
    f = theano.function([x], pre_softmax)
    img = np.array(Image.open('bike.jpg'))
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1) / 255.0
    img = img[np.newaxis].astype('float32')
    res = f(img)
    print _class_names[np.argmax(res)]


def apply_vgg(x):
    from sklearn_theano.feature_extraction.caffe.vgg import create_theano_expressions
    mean_values = np.array([104, 117, 123]).reshape((3, 1, 1))
    # Convert RGB to BGR
    xx = x[:, ::-1, :, :] * 255.0
    xx = xx - mean_values[np.newaxis, :, :, :].astype('float32')

    net = create_theano_expressions(inputs=('data', xx))
    pre_softmax = net[0]['prob']

    return pre_softmax.flatten(2)


def test_vgg():
    x = T.tensor4('x')
    pre_softmax = apply_vgg(x)
    f = theano.function([x], pre_softmax)
    img = np.array(Image.open('pen.jpg'))
    img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1) / 255.0
    img = img[np.newaxis].astype('float32')
    res = f(img)
    print _class_names[np.argmax(res)]

test_vgg()
