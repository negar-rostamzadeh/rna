import logging
import numpy as np
import theano
import theano.tensor as T
from blocks.bricks.base import application
from blocks.bricks.interfaces import Activation
from blocks.extensions import SimpleExtension
from blocks.roles import add_role
from blocks.roles import AuxiliaryRole
from blocks.graph import ComputationGraph
from blocks.initialization import NdarrayInitialization
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger('main.utils')


class HardTanh(Activation):
    @application(inputs=['input_'], outputs=['output'])
    def apply(self, input_):
        return T.minimum(T.maximum(input_, -1), 1)


class BnParamRole(AuxiliaryRole):
    pass
BNPARAM = BnParamRole()


def shared_param(init, name, cast_float32, role, **kwargs):
    if cast_float32:
        v = np.float32(init)
    p = theano.shared(v, name=name, **kwargs)
    add_role(p, role)
    return p


class LRDecay(SimpleExtension):
    def __init__(self, lr_var, lrs, until_which_epoch, **kwargs):
        super(LRDecay, self).__init__(**kwargs)
        # assert self.num_epochs == until_which_epoch[-1]
        self.iter = 0
        self.lrs = lrs
        self.until_which_epoch = until_which_epoch
        self.lr_var = lr_var

    def do(self, which_callback, *args):
        self.iter += 1
        if self.iter < self.until_which_epoch[-1]:
            lr_index = [self.iter < epoch for epoch
                        in self.until_which_epoch].index(True)
        else:
            print "WARNING: the smallest learning rate is using."
            lr_index = -1
        self.lr_var.set_value(np.float32(self.lrs[lr_index]))


class ErrorPerVideo(SimpleExtension):
    def __init__(self, model, **kwargs):
        super(ErrorPerVideo, self).__init__(**kwargs)
        self.model = model

    def do(self, which_callback, *args):
        import ipdb; ipdb.set_trace()
        vds = self.main_loop.extensions[1].data_stream
        num_batches = 1 + vds.data_stream.dataset.num_examples / vds.batch_size
        # for i in range(9):
        #     batch = vds.get_epoch_iterator().next()
        # import ipdb; ipdb.set_trace()

        mlp = self.main_loop.model.top_bricks[1]
        probs = mlp.apply_outputs
        ComputationGraph(probs).inputs


class AttributeDict(dict):
    __getattr__ = dict.__getitem__

    def __setattr__(self, a, b):
        self.__setitem__(a, b)


class SaveParams(SimpleExtension):
    """Finishes the training process when triggered."""
    def __init__(self, early_stop_var, model, save_path, **kwargs):
        super(SaveParams, self).__init__(**kwargs)
        self.early_stop_var = early_stop_var
        self.save_path = save_path
        params_dicts = model.get_parameter_dict()
        self.params_names = params_dicts.keys()
        self.params_values = params_dicts.values()
        self.to_save = {}
        self.best_value = None
        self.add_condition(('after_training',), self.save)
        self.add_condition(('on_interrupt',), self.do)
        self.add_condition(('after_epoch',), self.do)

    def save(self, which_callback, *args):
        to_save = {}
        for p_name, p_value in zip(self.params_names, self.params_values):
            to_save[p_name] = p_value.get_value()
        path = self.save_path + '/trained_params'
        np.savez_compressed(path, **to_save)

    def do(self, which_callback, *args):
        val = self.main_loop.log.current_row[self.early_stop_var]
        if self.best_value is None or val < self.best_value:
            self.best_value = val
            to_save = {}
            for p_name, p_value in zip(self.params_names, self.params_values):
                to_save[p_name] = p_value.get_value()
            path = self.save_path + '/trained_params_best'
            np.savez_compressed(path, **to_save)
        self.main_loop.log.current_row[
            'best_' + self.early_stop_var] = self.best_value


class SaveLog(SimpleExtension):
    def __init__(self, show=None, **kwargs):
        super(SaveLog, self).__init__(**kwargs)

    def do(self, which_callback, *args):
        epoch = self.main_loop.status['epochs_done']
        current_row = self.main_loop.log.current_row
        logger.info("\nIter:%d" % epoch)
        for element in current_row:
            logger.info(str(element) + ":%f" % current_row[element])


def apply_act(input, act_name):
    if input is None:
        return input
    act = {
        'relu': lambda x: T.maximum(0, x),
        'leakyrelu': lambda x: T.switch(x > 0., x, 0.1 * x),
        'linear': lambda x: x,
        'softplus': lambda x: T.log(1. + T.exp(x)),
        'sigmoid': lambda x: T.nnet.sigmoid(x),
        'softmax': lambda x: T.nnet.softmax(x),
    }.get(act_name)
    if act_name == 'softmax':
        input = input.flatten(2)
    return act(input)


class Glorot(NdarrayInitialization):
    def generate(self, rng, shape):
        if len(shape) == 2:
            input_size, output_size = shape
            high = np.sqrt(6) / np.sqrt(input_size + output_size)
            m = rng.uniform(-high, high, size=shape)
            if shape == (128, 512):
                high = np.sqrt(6) / np.sqrt(256)
                mi = rng.uniform(-high, high, size=(128, 128))
                mf = rng.uniform(-high, high, size=(128, 128))
                mc = rng.uniform(-high, high, size=(128, 128))  # np.identity(128) * 0.95
                mo = rng.uniform(-high, high, size=(128, 128))
                m = np.hstack([mi, mf, mc, mo])
            else:
                import ipdb
                ipdb.set_trace
        elif len(shape) == 4:
            fan_in = np.prod(shape[1:])
            poolsize = (2, 2)
            fan_out = (shape[0] * np.prod(shape[2:]) / np.prod(poolsize))
            high = np.sqrt(6) / np.sqrt(fan_in + fan_out)
            m = rng.uniform(-high, high, size=shape)
        return m.astype(theano.config.floatX)


def to_rgb1(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.float32)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret


def visualize_attention(model, configs, eval_function):
    cropper = model.children[0]
    location = T.fmatrix()
    scale = T.fmatrix()
    alpha = T.fmatrix()
    x = T.fvector()

    min_scale = [(float(configs['patch_shape'][0]) /
                  configs['cropper_input_shape'][0]),
                 (float(configs['patch_shape'][1]) /
                  configs['cropper_input_shape'][1])]

    re_loc = (
        (location + T.ones_like(location)) *
        np.array([configs['cropper_input_shape'][0] * 0.4,
                  configs['cropper_input_shape'][1] * 0.4]).astype('float32') +
        np.array([configs['cropper_input_shape'][0] * 0.1,
                  configs['cropper_input_shape'][1] * 0.1]).astype('float32'))

    scale2d = T.concatenate([scale, scale], axis=1)
    alpha2d = T.concatenate([alpha, alpha], axis=1)

    re_scale = (
        (scale2d + T.ones_like(scale2d)) *
        np.array([
            (1.1 - min_scale[0]) / 2.0,
            (1.1 - min_scale[1]) / 2.0]).astype('float32') +
        np.array(min_scale).astype('float32'))

    re_alpha = (
        (alpha2d + T.ones_like(alpha2d)) *
        np.array([0.98 / 2.0 + 0.001,
                  0.98 / 2.0 + 0.001]).astype('float32'))

    patch, mat, _ = cropper.apply(
        x.reshape((configs['batch_size'],
                   configs['num_channels'],) + configs['cropper_input_shape']),
        np.array([list(configs['cropper_input_shape'])]),
        re_loc,
        re_scale,
        re_alpha)
    grads = T.grad(T.mean(patch ** 2), x).reshape(
        (configs['batch_size'], configs['num_channels'],) + configs['cropper_input_shape'])
    get_patch = theano.function([x, location, scale, alpha],
                                [patch, grads],
                                allow_input_downcast=True)

    _, ds = configs['get_streams'](configs['batch_size'])
    data = ds.get_epoch_iterator().next()

    # T x B x 1 x X x Y
    frames = data[0]
    inps = ComputationGraph(model.location).inputs
    f = theano.function(inps, [model.location, model.scale, model.alpha])

    # T x B x 2or1
    locations, scales, alphas = f(frames)

    predictions = np.argmax(
        eval_function(data[0], data[1])[1], axis=1)

    # over time
    for t in range(locations.shape[0]):
        frame = frames[t] + ds.mean[np.newaxis]
        patches, _ = get_patch(
            frame.flatten(),
            locations[t], scales[t], alphas[t])
        _, grads_ = get_patch(
            np.ones(frame.flatten().shape),
            locations[t], scales[t], alphas[t])
        grads_ = (grads_ / np.max(grads_))
        # [:, np.newaxis]
        # grads_ = np.concatenate([grads_, grads_, grads_], axis=1)
        grads_[:, 0] += frame[:, 0]
        # grads_[:, 0] = grads_[:, 0] / np.max(grads_[:, 0], axis=0, keepdims=True)
        grads_[:, 1] = frame[:, 1]
        grads_[:, 2] = frame[:, 2]

        for exp in range(30):
            plt.figure()

            plt.subplot(121)
            img = grads_[exp]
            img = np.swapaxes(img[:, :, :, np.newaxis], 0, 3)[0]
            plt.imshow(img, interpolation='nearest')

            plt.subplot(122)
            img = patches[exp]
            img = np.swapaxes(img[:, :, :, np.newaxis], 0, 3)[0]
            plt.imshow(img, interpolation='nearest')

            plt.tight_layout()
            good = predictions[exp] == data[1][exp]
            plt.savefig('sample_e' + str(exp) + '_t' + str(t) +
                        '_' + str(good) + '.png')
            print 'sample_e' + str(exp) + '_t' + str(t) + '.png saved!'
