import theano
import theano.tensor as T
from crop import LocallySoftRectangularCropper
from crop import Gaussian
import numpy as np
from datasets import get_cooking_streams
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def draw(img):
    plt.imshow(np.swapaxes(img[:, :, :, np.newaxis], 0, 3)[0],
               interpolation='nearest')

tds, _ = get_cooking_streams(1)
# from datasets import get_bmnist_streams
# tds, _ = get_bmnist_streams(1)
res = tds.get_epoch_iterator(as_dict=True).next()['features']
# shape: 3 x 125 x 200
img = res[5, 0]
draw(img)
plt.savefig('img.png')

location = T.fmatrix()
scale = T.fmatrix()
alpha = T.fmatrix()
x = T.fvector()
batch_size = 1
num_channel = 3
patch_shape = (28, 28)
image_shape = (125, 200)
hyperparameters = {}
hyperparameters["cutoff"] = 3000
hyperparameters["batched_window"] = True

cropper = LocallySoftRectangularCropper(
    patch_shape=patch_shape,
    hyperparameters=hyperparameters,
    kernel=Gaussian())

patch1, matrix, dx2 = cropper.apply(
    x.reshape((batch_size, num_channel,) + image_shape),
    np.array([list(image_shape)]),
    location,
    scale,
    alpha)
grads = T.grad(T.mean(patch1), x)
grad_scale = abs(T.grad(T.mean(patch1), scale))
grad_location = abs(T.grad(T.mean(patch1), location))
grad_alpha = abs(T.grad(T.mean(patch1), alpha))

f = theano.function(
    [x, location, scale, alpha],
    [patch1, grads, grad_scale + grad_location + grad_alpha, matrix, dx2],
    allow_input_downcast=True)

image = img.flatten().astype('float32')
locations = [[60, 50], [100, 70], [10, 190]]
scales = [[0.224, 0.14], [0.6, 0.6], [1, 1]]
alphas = [[0.001, 0.001], [0.5, 0.5], [0.9, 0.9]]

for i in np.arange(3):
    for j in np.arange(3):
        for k in np.arange(3):
            location_ = [locations[i]]
            scale_ = [scales[j]]
            alpha_ = [alphas[k]]

            plt.figure()

            plt.subplot(131)
            p = f(image, location_, scale_, alpha_)[0][0]
            # import ipdb; ipdb.set_trace()
            draw(p)

            plt.subplot(132)
            g = np.abs(f(image, location_, scale_, alpha_)[1].reshape(
                3, image_shape[0], image_shape[1]))
            g = np.sum(g, axis=0)
            g = (g - g.mean()) / g.std()
            plt.imshow(g[1:-1, 1:-1], cmap=plt.get_cmap('gray'), interpolation='nearest')

            plt.subplot(133)
            m = np.abs(f(image, location_, scale_, alpha_)[3][0])
            res = f(image, location_, scale_, alpha_)[4]
            plt.imshow(m, interpolation='nearest', vmin=0, vmax=1)
            plt.tight_layout()
            plt.savefig('sample' + str(i) + str(j) + str(k) + '.png', dpi=450)

            print np.abs(f(image, location_, scale_, alpha_)[2])
