import logging
import os
import sys
import time
import numpy as np
import theano.tensor as T
from theano import config
import theano
from blocks.algorithms import (GradientDescent, Adam,
                               CompositeRule, StepClipping)
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.bricks.cost import BinaryCrossEntropy, MisclassificationRate
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.bricks import Rectifier, Logistic, MLP
from blocks.main_loop import MainLoop
from blocks.model import Model
from utils import SaveLog, SaveParams, Glorot, visualize_attention, LRDecay
from utils import plot_curves
from blocks.initialization import Constant
from blocks.graph import ComputationGraph
from LSTM_attention_model import LSTMAttention
from blocks.monitoring import aggregation
floatX = theano.config.floatX
logger = logging.getLogger('main')


def setup_model(configs):

    tensor5 = theano.tensor.TensorType(config.floatX, (False,) * 5)
    # shape: T x B x C x X x Y
    input_ = tensor5('features')
    # shape: B x Classes
    target = T.lmatrix('targets')

    model = LSTMAttention(
        configs,
        weights_init=Glorot(),
        biases_init=Constant(0))
    model.initialize()

    (h, c, location, scale, patch, downn_sampled_input,
        conved_part_1, conved_part_2, pre_lstm) = model.apply(input_)

    classifier = MLP(
        [Rectifier(), Logistic()],
        configs['classifier_dims'],
        weights_init=Glorot(),
        biases_init=Constant(0))
    classifier.initialize()

    probabilities = classifier.apply(h[-1])
    cost = BinaryCrossEntropy().apply(target, probabilities)
    cost.name = 'CE'
    error_rate = MisclassificationRate().apply(target, probabilities)
    error_rate.name = 'ER'
    model.cost = cost

    if configs['load_pretrained']:
        blocks_model = Model(model.cost)
        all_params = blocks_model.parameters
        with open('VGG_CNN_params.npz') as f:
            loaded = np.load(f)
            all_conv_params = loaded.keys()
            for param in all_params:
                if param.name in loaded.keys():
                    assert param.get_value().shape == loaded[param.name].shape
                    param.set_value(loaded[param.name])
                    all_conv_params.pop(all_conv_params.index(param.name))
        print "the following parameters did not match: " + str(all_conv_params)

    if configs['test_model']:
        cg = ComputationGraph(model.cost)
        f = theano.function(cg.inputs, [model.cost],
                            on_unused_input='ignore',
                            allow_input_downcast=True)
        data = np.random.randn(10, 40, 3, 224, 224)
        targs = np.random.randn(40, 101)
        f(data, targs)
        print "Test passed! ;)"

    model.monitorings = [cost, error_rate]

    return model


def train(model, get_streams, save_path, num_epochs,
          batch_size, lrs, until_which_epoch, grad_clipping):
    monitorings = model.monitorings

    # Training
    blocks_model = Model(model.cost)
    all_params = blocks_model.parameters
    print "Number of found parameters:" + str(len(all_params))
    print all_params

    default_lr = np.float32(1e-4)
    lr_var = theano.shared(default_lr, name="learning_rate")

    clipping = StepClipping(threshold=np.cast[floatX](grad_clipping))
    # sgd_momentum = Momentum(
    #     learning_rate=0.0001,
    #     momentum=0.95)
    # step_rule = CompositeRule([clipping, sgd_momentum])
    adam = Adam(learning_rate=lr_var)
    step_rule = CompositeRule([clipping, adam])
    training_algorithm = GradientDescent(
        cost=model.cost, parameters=all_params,
        step_rule=step_rule)

    monitored_variables = [
        lr_var,
        aggregation.mean(training_algorithm.total_gradient_norm)] + monitorings

    for param in all_params:
        name = param.name
        to_monitor = training_algorithm.gradients[param].norm(2)
        to_monitor.name = name + "_grad_norm"
        monitored_variables.append(to_monitor)
        to_monitor = param.norm(2)
        to_monitor.name = name + "_norm"
        monitored_variables.append(to_monitor)

    train_data_stream, valid_data_stream = get_streams(batch_size)

    train_monitoring = TrainingDataMonitoring(
        variables=monitored_variables,
        prefix="train",
        after_epoch=True)

    valid_monitoring = DataStreamMonitoring(
        variables=monitored_variables,
        data_stream=valid_data_stream,
        prefix="valid",
        after_epoch=True)

    main_loop = MainLoop(
        algorithm=training_algorithm,
        data_stream=train_data_stream,
        model=blocks_model,
        extensions=[
            train_monitoring,
            valid_monitoring,
            FinishAfter(after_n_epochs=num_epochs),
            SaveParams('valid_CE',
                       blocks_model, save_path),
            SaveLog(save_path, after_epoch=True),
            ProgressBar(),
            LRDecay(lr_var, lrs, until_which_epoch,
                    after_epoch=True),
            Printing(after_epoch=True)])
    main_loop.run()


def evaluate(model, load_path, plot):
    with open(load_path + 'trained_params_best.npz') as f:
        loaded = np.load(f)
        blocks_model = Model(model.cost)
        params_dicts = blocks_model.get_parameter_dict()
        params_names = params_dicts.keys()
        for param_name in params_names:
            param = params_dicts[param_name]
            # '/f_6_.W' --> 'f_6_.W'
            slash_index = param_name.find('/')
            param_name = param_name[slash_index + 1:]
            assert param.get_value().shape == loaded[param_name].shape
            param.set_value(loaded[param_name])

    if plot:
        train_data_stream, valid_data_stream = get_streams(20)
        # T x B x F
        data = train_data_stream.get_epoch_iterator().next()
        cg = ComputationGraph(model.cost)
        f = theano.function(cg.inputs, [model.location, model.scale],
                            on_unused_input='ignore',
                            allow_input_downcast=True)
        res = f(data[1], data[0])
        for i in range(10):
            visualize_attention(data[0][:, i, :],
                                res[0][:, i, :], res[1][:, i, :],
                                image_shape=(512, 512), prefix=str(i))

        plot_curves(path=load_path,
                    to_be_plotted=['train_categoricalcrossentropy_apply_cost',
                                   'valid_categoricalcrossentropy_apply_cost'],
                    yaxis='Cross Entropy',
                    titles=['train', 'valid'],
                    main_title='CE')

        plot_curves(path=load_path,
                    to_be_plotted=['train_learning_rate',
                                   'train_learning_rate'],
                    yaxis='lr',
                    titles=['train', 'train'],
                    main_title='lr')

        plot_curves(path=load_path,
                    to_be_plotted=['train_total_gradient_norm',
                                   'valid_total_gradient_norm'],
                    yaxis='GradientNorm',
                    titles=['train', 'valid'],
                    main_title='GradientNorm')

        for grad in ['_total_gradient_norm',
                     '_total_gradient_norm',
                     '_/lstmattention.W_patch_grad_norm',
                     '_/lstmattention.W_state_grad_norm',
                     '_/lstmattention.initial_cells_grad_norm',
                     '_/lstmattention.initial_location_grad_norm',
                     '_/lstmattention/lstmattention_mlp/linear_0.W_grad_norm',
                     '_/lstmattention/lstmattention_mlp/linear_1.W_grad_norm',
                     '_/mlp/linear_0.W_grad_norm',
                     '_/mlp/linear_1.W_grad_norm']:
            plot_curves(path=load_path,
                        to_be_plotted=['train' + grad,
                                       'valid' + grad],
                        yaxis='GradientNorm',
                        titles=['train',
                                'valid'],
                        main_title=grad.replace(
                            "_", "").replace("/", "").replace(".", ""))

        plot_curves(path=load_path,
                    to_be_plotted=[
                        'train_misclassificationrate_apply_error_rate',
                        'valid_misclassificationrate_apply_error_rate'],
                    yaxis='Error rate',
                    titles=['train', 'valid'],
                    main_title='Error')
        print 'plot printed'


if __name__ == "__main__":
        dataset = str(sys.argv[1])
        available_datasets = ['cmv', 'ucf101', 'cooking']
        if dataset not in available_datasets:
            print 'ERROR: available datasets are: ' + str(available_datasets)
            sys.exit()
        logging.basicConfig(level=logging.INFO)

        configs = {}
        if dataset == 'cmv':
            from datasets import get_cmv_v1_streams
            configs['get_streams'] = get_cmv_v1_streams
            configs['save_path'] = 'results/CMV_500_epochs'
            configs['num_epochs'] = 500
            configs['batch_size'] = 100
            configs['lrs'] = [1e-4, 1e-5]
            configs['until_which_epoch'] = [30, configs['num_epochs']]
            configs['grad_clipping'] = 5
            configs['conv_layers'] = []
            configs['num_layers_first_half_of_conv'] = 0
            configs['fc_layers'] = [['fc', (576, 256), 'relu']]
            configs['lstm_dim'] = 256
            configs['attention_mlp_hidden_dims'] = [128]
            configs['cropper_input_shape'] = (100, 100)
            configs['patch_shape'] = (24, 24)
            configs['num_channels'] = 1
            configs['classifier_dims'] = [configs['lstm_dim'], 128, 10]
            configs['load_pretrained'] = False

        elif dataset == 'ucf101':
            from datasets import get_ucf_streams
            configs['get_streams'] = get_ucf_streams
            configs['save_path'] = 'results/UCF_500_epochs'
            configs['num_epochs'] = 500
            configs['batch_size'] = 40
            configs['lrs'] = [1e-4, 1e-5]
            configs['until_which_epoch'] = [30, configs['num_epochs']]
            configs['grad_clipping'] = 5
            configs['conv_layers'] = [                            # 3   x 224 x 224 --> 3 x 14 x 14
                ['conv_1_1', (64, 3, 3, 3), None, None],          # 64  x 14  x 14
                ['conv_1_2', (64, 64, 3, 3), None, None],         # 64  x 14  x 14
                # ['conv_1_2', (64, 64, 3, 3), (2, 2), None],
                ['conv_2_1', (128, 64, 3, 3), None, None],        # 128 x 14  x 14
                ['conv_2_2', (128, 128, 3, 3), None, None],       # 128 x 14  x 14
                # ['conv_2_2', (128, 128, 3, 3), (2, 2), None],
                ['conv_3_1', (256, 128, 3, 3), None, None],       # 256 x 14  x 14
                ['conv_3_2', (256, 256, 3, 3), None, None],       # 256 x 14  x 14
                ['conv_3_3', (256, 256, 3, 3), None, None],       # 256 x 14  x 14
                # ['conv_3_3', (256, 256, 3, 3), (2, 2), None],   # 256 x 14  x 14
                ['conv_4_1', (512, 256, 3, 3), None, None],       # 512 x 14  x 14
                ['conv_4_2', (512, 512, 3, 3), None, None],       # 512 x 14  x 14
                ['conv_4_3', (512, 512, 3, 3), None, None],       # 512 x 14  x 14
                # ['conv_4_3', (512, 512, 3, 3), (2, 2), None],
                ['conv_5_1', (512, 512, 3, 3), None, None],       # 512 x 14  x 14
                ['conv_5_2', (512, 512, 3, 3), None, None],       # 512 x 14  x 14
                ['conv_5_3', (512, 512, 3, 3), (2, 2), None]]     # 512 x 7   x 7
            configs['num_layers_first_half_of_conv'] = 0
            configs['fc_layers'] = [['fc6', (25088, 4096), 'relu'],
                                    ['fc7', (4096, 4096), 'relu'],
                                    ['fc8-1', (4096, 101), 'lin']]
            configs['lstm_dim'] = 256
            configs['attention_mlp_hidden_dims'] = [128]
            configs['cropper_input_shape'] = (224, 224)
            configs['patch_shape'] = (14, 14)
            configs['num_channels'] = 3
            configs['classifier_dims'] = [configs['lstm_dim'], 128, 101]
            configs['load_pretrained'] = True

        elif dataset == 'cooking':
            from datasets import get_cooking_streams
            configs['get_streams'] = get_cooking_streams
            configs['save_path'] = 'results/Cook_500_epochs'
            configs['num_epochs'] = 500
            configs['batch_size'] = 40
            configs['lrs'] = [1e-4, 1e-5]
            configs['until_which_epoch'] = [30, configs['num_epochs']]
            configs['grad_clipping'] = 5
            configs['conv_layers'] = [                                       # 3   x 512 x 512
                ['conv_1_1', (64, 3, 3, 3), None, None],          # 64  x 512 x 512
                ['conv_1_2', (64, 64, 3, 3), (2, 2), None],       # 64  x 256 x 256
                ['conv_2_1', (128, 64, 3, 3), None, None],        # 128 x 256 x 256
                ['conv_2_2', (128, 128, 3, 3), (2, 2), None],     # 128 x 128 x 128
                ['conv_3_1', (256, 128, 3, 3), None, None],       # 256 x 128 x 128
                ['conv_3_2', (256, 256, 3, 3), None, None],       # 256 x 128 x 128
                ['conv_3_3', (256, 256, 3, 3), None, None],       # 256 x 128 x 128 --> 256 x 28 x 28
                # ['conv_3_3', (256, 256, 3, 3), (2, 2), None],
                ['conv_4_1', (512, 256, 3, 3), None, None],       # 512 x 28  x 28
                ['conv_4_2', (512, 512, 3, 3), None, None],       # 512 x 28  x 28
                ['conv_4_3', (512, 512, 3, 3), (2, 2), None],     # 512 x 14  x 14
                ['conv_5_1', (512, 512, 3, 3), None, None],       # 512 x 14  x 14
                ['conv_5_2', (512, 512, 3, 3), None, None],       # 512 x 14  x 14
                ['conv_5_3', (512, 512, 3, 3), (2, 2), None]]     # 512 x 7   x 7
            configs['num_layers_first_half_of_conv'] = 7
            configs['fc_layers'] = [['fc6', (25088, 4096), 'relu'],
                                    ['fc7', (4096, 4096), 'relu']]
            configs['lstm_dim'] = 256
            configs['attention_mlp_hidden_dims'] = [128]
            configs['cropper_input_shape'] = (128, 128)
            configs['patch_shape'] = (28, 28)
            configs['num_channels'] = 3
            configs['classifier_dims'] = [configs['lstm_dim'], 128, 10]
            configs['load_pretrained'] = True

        configs['test_model'] = True
        timestr = time.strftime("%Y_%m_%d_at_%H_%M")
        save_path = configs['save_path'] + '_' + timestr
        log_path = os.path.join(save_path, 'log.txt')
        os.makedirs(save_path)
        fh = logging.FileHandler(filename=log_path)
        fh.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        for item in configs:
            logger.info(item + ': %s' % str(configs[item]))

        model = setup_model(configs)

        eval_ = False
        if eval_:
            evaluate(model, 'results/test_2015_11_12_at_20_35/', plot=False)
            ds, _ = configs['get_streams'](20)
            data = ds.get_epoch_iterator(as_dict=True).next()
            inputs = ComputationGraph(model.patch).inputs
            f = theano.function(inputs,
                                [model.location, model.scale,
                                 model.patch, model.downn_sampled_input])
            res = f(data['features'])
            location, scale, patch, downn_sampled_input = res
            # os.makedirs('res_frames/')
            # os.makedirs('res_patch/')
            # os.makedirs('res_downn_sampled_input/')
            # for i, f in enumerate(data['features']):
            #     plt.imshow(f[0].reshape(100, 100), cmap=plt.gray(),
            #                interpolation='nearest')
            #     plt.savefig('res_frames/img_' + str(i) + '.png')
            # for i, p in enumerate(patch):
            #     plt.imshow(p[0, 0], cmap=plt.gray(), interpolation='nearest')
            #     plt.savefig('res_patch/img_' + str(i) + '.png')
            # for i, d in enumerate(downn_sampled_input):
            #     plt.imshow(d[0, 0], cmap=plt.gray(), interpolation='nearest')
            #     plt.savefig('res_downn_sampled_input/img_' + str(i) + '.png')

            for i in range(10):
                visualize_attention(data['features'][:, i],
                                    (location[:, i] + 1) * 512 / 2,
                                    scale[:, i] + 1 + 0.24,
                                    image_shape=(512, 512), prefix=str(i))
        else:
            # evaluate(model, 'results/v2_len10_mlp_2015_11_13_at_18_37/',
            #          plot=False)
            train(model, configs)
