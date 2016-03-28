import theano
import theano.tensor as T
from theano import config
from crop import LocallySoftRectangularCropper
from crop import Gaussian
import numpy as np
from datasets import get_bmnist_streams
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


tensor5 = T.TensorType(config.floatX, (False,) * 4)
# shape: B x C x X x Y
input_ = tensor5('features')

cropper = LocallySoftRectangularCropper(
    patch_shape=(28, 28),
    hyperparameters={'cutoff': 3000, 'batched_window': True},
    kernel=Gaussian())

down, W, dx2 = cropper.apply(
    input_,
    np.array([list((100, 100))]),
    T.constant(
        99 *
        [[80, 70]]).astype('float32'),
    T.constant(
        99 *
        [[0.28, 0.28]]).astype('float32'),
    T.constant(
        99 *
        [[0.001, ] * 2]).astype('float32'))

f = theano.function([input_], [down, W, dx2])

data = get_bmnist_streams(99)[0].get_epoch_iterator().next()
res = f(data[0][0])

print np.min(res[2][0], axis=0)
print np.sum(res[1][0], axis=0)
plt.imshow(res[1][0], interpolation='nearest')
plt.savefig('w.png')


import ipdb; ipdb.set_trace()
