from blocks.bricks import Initializable, Tanh, Rectifier
from blocks.bricks.base import application
from blocks.roles import add_role, WEIGHT, BIAS, INITIAL_STATE
from blocks.utils import shared_floatx_nans, shared_floatx_zeros
from blocks.bricks.recurrent import BaseRecurrent, recurrent
from theano.tensor.signal.downsample import max_pool_2d
from theano.sandbox.cuda.dnn import dnn_conv
import theano.tensor as tensor
import numpy as np
from crop import LocallySoftRectangularCropper
from crop import Gaussian


class LSTMAttention(BaseRecurrent, Initializable):
    def __init__(self, configs, **kwargs):
        super(LSTMAttention, self).__init__(**kwargs)
        self.attention_mlp_hidden_dims =\
            configs['attention_mlp_hidden_dims'] + [4]
        self.conv_layers = configs['conv_layers']
        self.fc_layers = configs['fc_layers']
        self.num_layers_first_half_of_conv =\
            configs['num_layers_first_half_of_conv']
        self.lstm_dim = configs['lstm_dim']
        self.cropper_input_shape = configs['cropper_input_shape']
        self.patch_shape = configs['patch_shape']
        self.batch_size = configs['batch_size']
        self.num_channels = configs['num_channels']
        cropper = LocallySoftRectangularCropper(
            patch_shape=self.patch_shape,
            hyperparameters={'cutoff': 3000, 'batched_window': True},
            kernel=Gaussian())
        self.min_scale = [
            (self.patch_shape[0] + 0.0) / self.cropper_input_shape[0],
            (self.patch_shape[1] + 0.0) / self.cropper_input_shape[1]]
        self.children = [cropper, Tanh(), Rectifier()]

    def get_dim(self, name):
        if name == 'inputs':
            return self.lstm_dim * 4
        if name in ['states', 'cells']:
            return self.lstm_dim
        if name in ['location', 'scale', 'alpha']:
            return 2
        if name == 'mask':
            return 0
        if name == 'patch':
            return np.prod(self.patch_shape)
        if name == 'downn_sampled_inputs':
            return np.prod(self.patch_shape)
        return super(LSTMAttention, self).get_dim(name)

    def apply_attention_mlp(self, x):
        tanh = self.children[1].apply
        relu = self.children[2].apply
        pre_1 = tensor.dot(x, self.w_1_mlp) + self.b_1_mlp
        act_1 = relu(pre_1)
        pre_2 = (tensor.dot(act_1, self.w_2_mlp) + self.b_2_mlp +
                 np.asarray([0.0, 0.0, -0.2, -0.2],
                            dtype='float32'))
        act_2 = tanh(pre_2)
        return act_2

    # input shape: B x C x X x Y
    # output shape: B x 1 x X x Y
    def rgb2gray(self, rgb):
        mat = tensor.ones_like(rgb)
        mat = tensor.concatenate([mat[:, 0:1, :, :] * 0.2989,
                                  mat[:, 1:2, :, :] * 0.5870,
                                  mat[:, 2:3, :, :] * 0.1140],
                                 axis=1)
        mat = mat.astype('float32')
        gray = tensor.sum(rgb * mat, axis=1)
        gray = gray.dimshuffle(0, 'x', 1, 2)
        return gray

    def apply_conv(self, x, conv_layers):
        out = x
        relu = self.children[2].apply
        for layer in conv_layers:
            name, _, pool_shape, pool_stride = layer
            print name
            w = {w.name: w for w in self.conv_ws}[name + '_w']
            b = {b.name: b for b in self.conv_bs}[name + '_b']
            conv = dnn_conv(out, w, border_mode='full')
            # m = conv.mean(0, keepdims=True)
            # s = conv.var(0, keepdims=True)
            # conv = (conv - m) / tensor.sqrt(s + np.float32(1e-8))
            conv = conv + b.dimshuffle('x', 0, 'x', 'x')
            out = relu(conv)
            if pool_shape is not None:
                out = max_pool_2d(
                    input=out, ds=pool_shape,
                    st=pool_stride, ignore_border=True)
        return out

    def apply_fc(self, x):
        out = x
        for layer in self.fc_layers:
            name, shape, act = layer
            w = {w.name: w for w in self.fc_ws}[name + '_w']
            b = {b.name: b for b in self.fc_bs}[name + '_b']
            if act == 'relu':
                act = self.children[2].apply
            elif act == 'tanh':
                act = self.children[1].apply
            elif act == 'lin':
                act = lambda n: n
            out = tensor.dot(out, w)
            # m = out.mean(0, keepdims=True)
            # s = out.var(0, keepdims=True)
            # out = (out - m) / tensor.sqrt(s + np.float32(1e-8))
            out = act(out + b)
        return out

    # image: B x C x X x Y
    def down_sampler(self, image):
        cropper = self.children[0]
        downn_sampled_inputs, _, _ = cropper.apply(
            image,
            np.array([list(self.cropper_input_shape)]),
            tensor.constant(
                self.batch_size *
                [[self.cropper_input_shape[0] / 2,
                  self.cropper_input_shape[1] / 2]]).astype('float32'),
            tensor.constant(
                self.batch_size *
                [self.min_scale]).astype('float32'),
            tensor.constant(
                self.batch_size *
                [[0.001, ] * 2]).astype('float32'))
        return downn_sampled_inputs

    def _allocate(self):
        self.conv_ws = []
        self.conv_bs = []
        for layer in self.conv_layers:
            name, filter_shape, pool_shape, pool_stride = layer
            self.conv_ws.append(shared_floatx_nans(
                filter_shape, name=name + '_w'))
            self.conv_bs.append(shared_floatx_nans(
                (filter_shape[0],), name=name + '_b'))
        self.fc_ws = []
        self.fc_bs = []
        for layer in self.fc_layers:
            name, shape, act = layer
            self.fc_ws.append(shared_floatx_nans(
                shape, name=name + '_w'))
            self.fc_bs.append(shared_floatx_nans(
                (shape[1],), name=name + '_b'))
        self.b_1_mlp = shared_floatx_nans(
            (self.attention_mlp_hidden_dims[0],), name='b_1_mlp')
        self.b_2_mlp = shared_floatx_nans(
            (self.attention_mlp_hidden_dims[1],), name='b_2_mlp')
        self.w_1_mlp = shared_floatx_nans(
            (np.prod(self.patch_shape) + self.lstm_dim + 4,
                self.attention_mlp_hidden_dims[0]), name='w_1_mlp')
        self.w_2_mlp = shared_floatx_nans(
            (self.attention_mlp_hidden_dims[0],
                self.attention_mlp_hidden_dims[1]), name='w_2_mlp')
        self.W_pre_lstm = shared_floatx_nans(
            (self.fc_layers[-1][1][1] + 4, 4 * self.lstm_dim),
            name='W_pre_lstm')
        self.b_pre_lstm = shared_floatx_nans((4 * self.lstm_dim,),
                                             name='b_pre_lstm')
        self.W_state = shared_floatx_nans((self.lstm_dim, 4 * self.lstm_dim),
                                          name='W_state')
        self.initial_state_ = shared_floatx_zeros((self.lstm_dim,),
                                                  name="initial_state")
        self.initial_cells = shared_floatx_zeros((self.lstm_dim,),
                                                 name="initial_cells")
        self.initial_location = shared_floatx_zeros((2,),
                                                 name="initial_location")
        self.initial_scale = shared_floatx_zeros((1,),
                                                 name="initial_scale")
        self.initial_alpha = shared_floatx_zeros((1,),
                                                 name="initial_alpha")
        add_role(self.W_state, WEIGHT)
        add_role(self.W_pre_lstm, WEIGHT)
        add_role(self.b_pre_lstm, BIAS)
        add_role(self.b_1_mlp, BIAS)
        add_role(self.b_2_mlp, BIAS)
        add_role(self.w_1_mlp, WEIGHT)
        add_role(self.w_2_mlp, WEIGHT)
        add_role(self.initial_state_, INITIAL_STATE)
        add_role(self.initial_cells, INITIAL_STATE)
        add_role(self.initial_location, INITIAL_STATE)
        add_role(self.initial_scale, INITIAL_STATE)
        add_role(self.initial_alpha, INITIAL_STATE)
        for w in self.conv_ws + self.fc_ws:
            add_role(w, WEIGHT)
        for b in self.conv_bs + self.fc_bs:
            add_role(b, BIAS)

        self.parameters = [
            self.W_state, self.W_pre_lstm, self.w_1_mlp, self.w_2_mlp,
            self.b_pre_lstm, self.b_1_mlp, self.b_2_mlp, self.initial_state_,
            self.initial_cells, self.initial_location, self.initial_scale,
            self.initial_alpha] +\
            self.conv_ws + self.conv_bs +\
            self.fc_ws + self.fc_bs

    def _initialize(self):
        for weights in self.parameters[:4] + self.conv_ws + self.fc_ws:
            self.weights_init.initialize(weights, self.rng)
        for biases in self.parameters[4:7] + self.conv_bs + self.fc_bs:
            self.biases_init.initialize(biases, self.rng)

    def test(self, inputs):
        conved_part_1 = self.apply_conv(
            inputs[0],
            conv_layers=self.conv_layers)
        flat_conved_part_1 = conved_part_1.flatten(2)
        pre_lstm = self.apply_fc(flat_conved_part_1)
        return pre_lstm

    @recurrent(sequences=['inputs', 'mask'],
               states=['states', 'cells', 'location', 'scale', 'alpha'],
               contexts=[],
               outputs=['states', 'cells', 'location', 'scale', 'alpha',
                        'patch', 'downn_sampled_inputs', 'conved_part_1',
                        'conved_part_2', 'pre_lstm'])
    def apply(self, inputs, states, cells, location, scale, alpha, mask=None):
        def slice_last(x, no):
            return x[:, no * self.lstm_dim: (no + 1) * self.lstm_dim]

        tanh = self.children[1].apply
        cropper = self.children[0]

        # inputs shape:  B x C x X x Y
        # outputs shape: B x C' x X' x Y'
        conved_part_1 = self.apply_conv(
            inputs,
            conv_layers=self.conv_layers[0:self.num_layers_first_half_of_conv])

        # inputs shape:  B x C x X x Y
        # outputs shape: B x 1 x X x Y
        if self.num_channels == 3:
            gray_scale_inputs = self.rgb2gray(inputs)
        else:
            gray_scale_inputs = inputs

        # inputs shape:  B x 1 x X x Y
        # outputs shape: B x 1 x X' x Y'
        downn_sampled_inputs = self.down_sampler(gray_scale_inputs)

        # shape: B x F
        flat_downn_sampled_inputs = downn_sampled_inputs.flatten(ndim=2)

        # inputs shape:  B x F'
        # outputs shape: B x 3
        mlp_output = self.apply_attention_mlp(
            tensor.concatenate(
                [flat_downn_sampled_inputs,
                 0.00001 * location,
                 0.00001 * scale,
                 0.00001 * alpha,
                 states], axis=1))
        location = mlp_output[:, 0:2]
        location.name = 'location'
        scale = mlp_output[:, 2:3]
        scale.name = 'scale'
        alpha = mlp_output[:, 3:]
        alpha.name = 'alpha'

        scale2d = tensor.concatenate([scale, scale], axis=1)
        alpha2d = tensor.concatenate([alpha, alpha], axis=1)

        # inputs shape:  B x C' x X' x Y'
        # outputs shape: B x C' x X'' x Y''
        loc_to_cropper = (
            (location + tensor.ones_like(location)) *
            np.array([self.cropper_input_shape[0] * 0.4,
                      self.cropper_input_shape[0] * 0.4]).astype('float32') +
            np.array([self.cropper_input_shape[0] * 0.1,
                      self.cropper_input_shape[0] * 0.1]).astype('float32'))

        scale_to_cropper = (
            (scale2d + tensor.ones_like(scale2d)) *
            np.array([
                (1.1 - self.min_scale[0]) / 2.0,
                (1.1 - self.min_scale[1]) / 2.0]).astype('float32') +
            np.array(self.min_scale).astype('float32'))

        alpha_to_cropper = (
            (alpha2d + tensor.ones_like(alpha2d)) *
            np.array([0.98 / 2.0 + 0.001,
                      0.98 / 2.0 + 0.001]).astype('float32'))

        patch, _, _ = cropper.apply(
            conved_part_1,
            np.array([list(self.cropper_input_shape)]),
            # 0.00001 * loc_to_cropper + locs,
            # 0.00001 * scale_to_cropper + 1.0 * tensor.ones_like(scale_to_cropper),
            # 0.00001 * alpha_to_cropper + 0.001 * tensor.ones_like(scale_to_cropper))
            loc_to_cropper,
            scale_to_cropper,
            alpha_to_cropper)
        patch.name = 'patch'

        conved_part_2 = self.apply_conv(
            patch,
            conv_layers=self.conv_layers[self.num_layers_first_half_of_conv:])
        flat_conved_part_2 = conved_part_2.flatten(2)

        pre_lstm = self.apply_fc(flat_conved_part_2)
        pre_lstm = tensor.concatenate(
            [pre_lstm, location, scale, alpha], axis=1)
        transformed_pre_lstm = tensor.dot(
            pre_lstm, self.W_pre_lstm) + self.b_pre_lstm

        activation = tensor.dot(states, self.W_state) + transformed_pre_lstm
        in_gate = tensor.nnet.sigmoid(slice_last(activation, 0))
        forget_gate_input = slice_last(activation, 1)
        forget_gate = tensor.nnet.sigmoid(forget_gate_input +
                                          tensor.ones_like(forget_gate_input))
        next_cells = (forget_gate * cells +
                      in_gate * tanh(slice_last(activation, 2)))
        out_gate = tensor.nnet.sigmoid(slice_last(activation, 3))
        next_states = out_gate * tanh(next_cells)

        if mask:
            next_states = (mask[:, None] * next_states +
                           (1 - mask[:, None]) * states)
            next_cells = (mask[:, None] * next_cells +
                          (1 - mask[:, None]) * cells)

        return (next_states, next_cells, location, scale, alpha, patch,
                downn_sampled_inputs, conved_part_1, conved_part_2, pre_lstm)

    @application(outputs=apply.states)
    def initial_states(self, batch_size, *args, **kwargs):
        return [tensor.repeat(self.initial_state_[None, :], batch_size, 0),
                tensor.repeat(self.initial_cells[None, :], batch_size, 0),
                tensor.repeat(self.initial_location[None, :], batch_size, 0),
                tensor.repeat(self.initial_scale[None, :], batch_size, 0),
                tensor.repeat(self.initial_alpha[None, :], batch_size, 0)]
