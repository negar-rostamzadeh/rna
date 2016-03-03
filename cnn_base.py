from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.cuda.dnn import dnn_conv
import theano.tensor as tensor
from blocks.bricks import Rectifier, Tanh
import numpy as np
import theano

seed = 1234
rng = np.random.RandomState(seed)
conv_ws = []
conv_bs = []


def give_me_a_param(shape, name):
    if len(shape) == 1:
        high = 0
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) == 4:
        fan_in = np.prod(shape[1:])
        poolsize = (2, 2)
        fan_out = (shape[0] * np.prod(shape[2:]) / np.prod(poolsize))

    high = np.sqrt(6) / np.sqrt(fan_in + fan_out)
    init = rng.uniform(-high, high, size=shape)
    param = theano.shared(np.asarray(
        init, dtype=theano.config.floatX),
        name=name)
    return param

for layer in conv_layers:
    name, filter_shape, pool_shape, pool_stride = layer
    conv_ws.append(give_me_a_param(filter_shape, name + '_w'))
    conv_ws.append(shared_floatx_nans(
        filter_shape, name=name + '_w'))
    conv_bs.append(shared_floatx_nans(
        (filter_shape[0],), name=name + '_b'))

fc_ws = []
fc_bs = []
for layer in fc_layers:
    name, shape, act = layer
    fc_ws.append(shared_floatx_nans(
        shape, name=name + '_w'))
    fc_bs.append(shared_floatx_nans(
        (shape[1],), name=name + '_b'))


def apply_conv(x, conv_layers, conv_ws, conv_bs):
    out = x
    relu = Rectifier().apply
    for layer in conv_layers:
        name, _, pool_shape, pool_stride = layer
        print name
        w = {w.name: w for w in conv_ws}[name + '_w']
        b = {b.name: b for b in conv_bs}[name + '_b']
        conv = dnn_conv(out, w, border_mode='full')
        conv = conv + b.dimshuffle('x', 0, 'x', 'x')
        out = relu(conv)
        if pool_shape is not None:
            out = max_pool_2d(
                input=out, ds=pool_shape,
                st=pool_stride, ignore_border=True)
    return out


def apply_fc(x, fc_layers, fc_ws, fc_bs):
    out = x
    for layer in fc_layers:
        name, shape, act = layer
        w = {w.name: w for w in fc_ws}[name + '_w']
        b = {b.name: b for b in fc_bs}[name + '_b']
        if act == 'relu':
            act = Rectifier().apply
        elif act == 'tanh':
            act = Tanh().apply
        elif act == 'lin':
            act = lambda n: n
        out = tensor.dot(out, w)
        out = act(out + b)
    return out


configs['conv_layers'] = [                          # 1  x 28  x 28
    ['conv_1', (20, 1, 5, 5), (2, 2), None],        # 20 x 16 x 16
    ['conv_2', (50, 20, 5, 5), (2, 2), None],       # 50 x 10 x 10
    ['conv_3', (80, 50, 3, 3), (2, 2), None]]       # 80 x 6 x 6
configs['num_layers_first_half_of_conv'] = 0
configs['fc_layers'] = [['fc', (2880, 128), 'relu']]
