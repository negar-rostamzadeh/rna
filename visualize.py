import numpy as np
import theano
import theano.tensor as T
from crop import LocallySoftRectangularCropper
from crop import Gaussian
from PIL import Image
im = np.array(
    Image.open("exp.png"))[:, :, 0].flatten().astype('float32') / 255.0
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def to_rgb1(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.float32)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret


# im: size 10000
# im_2: size 100 x 100 x 3
def show_patch_on_frame(im, location_, scale_,
                        image_shape=(100, 100), patch_shape=(24, 24)):
    img = to_rgb1(im.reshape(image_shape))
    im_2 = img + np.zeros(img.shape)
    x_0 = (image_shape[0] / 2) - (patch_shape[0] / 2) / scale_[0]
    x_1 = (image_shape[0] / 2) + (patch_shape[0] / 2) / scale_[0]
    y_0 = (image_shape[1] / 2) - (patch_shape[1] / 2) / scale_[1]
    y_1 = (image_shape[1] / 2) + (patch_shape[1] / 2) / scale_[1]
    if x_0 < 0:
        x_0 = 0.0
    if x_1 > image_shape[0]:
        x_1 = image_shape[0]
    if y_0 < 0:
        y_0 = 0.0
    if y_1 > image_shape[1]:
        y_1 = image_shape[1]
    im_2[x_0:x_1, y_0:y_1, 0] = 1

    margin_0 = 1 + int(1 / scale_[0])
    margin_1 = 1 + int(1 / scale_[1])
    inner = img[margin_0 + x_0: -margin_0 + x_1,
                margin_1 + y_0: -margin_1 + y_1, 0]

    im_2[margin_0 + x_0: -margin_0 + x_1,
         margin_1 + y_0: -margin_1 + y_1, 0] = inner

    return im_2


# ims: size times x 100000
def show_patches_on_frames(ims, locations_, scales_,
                           image_shape=(100, 100), patch_shape=(24, 24)):
    hyperparameters = {}
    hyperparameters["cutoff"] = 3
    hyperparameters["batched_window"] = True
    location = T.fmatrix()
    scale = T.fmatrix()
    x = T.fvector()
    cropper = LocallySoftRectangularCropper(
        patch_shape=patch_shape,
        hyperparameters=hyperparameters,
        kernel=Gaussian())
    patch = cropper.apply(
        x.reshape((1, 1,) + image_shape),
        np.array([list(image_shape)]),
        location,
        scale)
    get_patch = theano.function([x, location, scale], patch,
                                allow_input_downcast=True)
    final_shape = (image_shape[0], image_shape[0] + patch_shape[0] + 5)
    ret = np.ones((ims.shape[0], ) + final_shape + (3,), dtype=np.float32)
    for i in range(ims.shape[0]):
        im = ims[i]
        location_ = locations_[i]
        scale_ = scales_[i]
        patch_on_frame = show_patch_on_frame(im, location_, scale_)
        ret[i, :, :image_shape[1], :] = patch_on_frame
        ret[i, -patch_shape[0]:, image_shape[1] + 5:, :] = to_rgb1(
            get_patch(im, [location_], [scale_])[0, 0])
        return ret


def save_files(frames, locations, scales):
    results = show_patches_on_frames(frames, locations, scales)
    for i, frame in enumerate(results):
        plt.imshow(
            frame,
            interpolation='nearest')
        plt.savefig('res/img_' + str(i) + '.png')
    print 'success!'
