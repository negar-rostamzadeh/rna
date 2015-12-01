import theano
import time
import theano.tensor as T
from crop import LocallySoftRectangularCropper
from crop import Gaussian
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

location = T.fmatrix()
scale = T.fmatrix()
x = T.fvector()
batch_size = 100
num_channel = 3
patch_shape = (14, 14)
image_shape = (224, 224)
hyperparameters = {}
hyperparameters["cutoff"] = 3
hyperparameters["batched_window"] = True

cropper = LocallySoftRectangularCropper(
    patch_shape=patch_shape,
    hyperparameters=hyperparameters,
    kernel=Gaussian())

patch = cropper.apply(
    x.reshape((batch_size, num_channel,) + image_shape),
    np.array([list(image_shape)]),
    location,
    scale)

f = theano.function([x, location, scale], patch, allow_input_downcast=True)

grad_l = T.grad(T.mean(patch), location)
grad_s = T.grad(T.mean(patch), scale)
g = theano.function([x, location, scale], [grad_l, grad_s], allow_input_downcast=True)

image = np.random.randn(
    np.prod(image_shape) * batch_size * num_channel).astype('float32')
location_ = [[60.0, 65.0]] * batch_size
scale_ = [[0.0625, 0.0625]] * batch_size

g_l, g_s = g(image, location_, scale_)
print np.mean(np.abs(g_l))
print np.mean(np.abs(g_s))
