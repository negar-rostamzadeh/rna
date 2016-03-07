import theano
import math
import theano.tensor as T
import numpy as np
from blocks.bricks import Brick, application
floatX = theano.config.floatX


def batched_tensordot(a, b, axes=2):
    return theano.tensor.basic._tensordot_as_dot(
        a, b, axes,
        dot=theano.sandbox.cuda.blas.batched_dot, batched=True)


class LocallySoftRectangularCropper(Brick):
    def __init__(self, patch_shape, kernel, hyperparameters, **kwargs):
        super(LocallySoftRectangularCropper, self).__init__(**kwargs)
        self.patch_shape = patch_shape
        self.kernel = kernel
        self.cutoff = hyperparameters["cutoff"]
        self.batched_window = hyperparameters["batched_window"]
        self.n_spatial_dims = len(patch_shape)

    def compute_crop_matrices(self, locations, scales, alphas, Is):
        Ws = []
        for axis in xrange(self.n_spatial_dims):
            n = T.cast(self.patch_shape[axis], 'float32')
            I = T.cast(Is[axis], 'float32').dimshuffle('x', 0, 'x')
            J = T.arange(n).dimshuffle('x', 'x', 0)
            location = locations[:, axis].dimshuffle(0, 'x', 'x')
            scale = scales[:, axis].dimshuffle(0, 'x', 'x')
            alpha = alphas[:, axis].dimshuffle(0, 'x', 'x')

            centered_J = (J - 0.5 * n) / scale
            max_val = n / (2.0 * scale)
            inv = (T.log(1.01 + (alpha * scale) * centered_J / max_val) -
                   T.log(1.01 - (alpha * scale) * centered_J / max_val)) / 2
            J = (inv * max_val) / (alpha * scale) + location

            dx2 = (I - J) ** 2
            # You know. Becuase of quantization, we don't get exactly zero
            # So let's make them zero!
            # import ipdb; ipdb.set_trace()
            # dx2 = dx2 - T.min(dx2, axis=1).flatten(2).dimshuffle(0, 'x', 1)
            # dx2 = T.round(dx2)
            W = self.kernel.density(dx2, scale, (alpha * scale), centered_J / max_val)
            # sumation = T.maximum(T.sum(W, axis=(1), keepdims=True), 2.0)
            # W = W / sumation
            Ws.append(W)

        return Ws, dx2

    def compute_hard_windows(self, image_shape, location, scale):
        # find topleft(front) and bottomright(back) corners for each patch
        a = location - 0.5 * (T.cast(self.patch_shape, theano.config.floatX) / scale)
        b = location + 0.5 * (T.cast(self.patch_shape, theano.config.floatX) / scale)

        # grow by three patch pixels
        a -= self.kernel.k_sigma_radius(self.cutoff, scale)
        b += self.kernel.k_sigma_radius(self.cutoff, scale)

        # clip to fit inside image and have nonempty window
        a = T.clip(a, 0, image_shape - 1)
        b = T.clip(b, a + 1, image_shape)

        if self.batched_window:
            # take the bounding box of all windows; now the slices
            # will have the same length for each sample and scan can
            # be avoided.  comes at the cost of typically selecting
            # more of the input.
            a = a.min(axis=0, keepdims=True)
            b = b.max(axis=0, keepdims=True)

        # make integer
        a = T.cast(T.floor(a), 'int16')
        b = T.cast(T.ceil(b), 'int16')

        return a, b

    @application(inputs="image image_shape location scale".split(),
                 outputs=['patch', 'matrix', 'dx2'])
    def apply(self, image, image_shape, location, scale, alpha):
        a, b = self.compute_hard_windows(image_shape, location, scale)

        patch, matrix, dx2 = self.apply_inner(
            image, location, scale, alpha, a[0], b[0])

        savings = (1 - T.cast((b - a).prod(axis=1), floatX) / image_shape.prod(axis=1))
        self.add_auxiliary_variable(savings, name="savings")

        return patch, matrix, dx2

    def apply_inner(self, image, location, scale, alpha, a, b):
        slices = [theano.gradient.disconnected_grad(T.arange(a[i], b[i]))
                  for i in xrange(self.n_spatial_dims)]
        hardcrop = image[
            np.index_exp[:, :] +
            tuple(slice(a[i], b[i])
                  for i in range(self.n_spatial_dims))]
        matrices, dx2 = self.compute_crop_matrices(location, scale, alpha, slices)
        patch = hardcrop
        for axis, matrix in enumerate(matrices):
            patch = batched_tensordot(patch, matrix, [[2], [1]])
        return patch, matrices[0], dx2


class Gaussian(object):
    def density(self, x2, scale, alpha, rng):
        sigma = self.sigma(scale, alpha, rng)
        volume = T.sqrt(2 * math.pi) * sigma
        return T.exp(-0.5 * x2 / (sigma ** 2)) / volume

    def sigma(self, scale, alpha, rng):
        tanhP = lambda x: 4 / (T.exp(x) + T.exp(-x)) ** 2
        sa = tanhP(rng * alpha / scale * 1.9)
        sigma = 0.5 / (scale * sa)
        return sigma

    def k_sigma_radius(self, k, scale):
        return k
