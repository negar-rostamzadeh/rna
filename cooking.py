import logging
import os
import time
import numpy as np
import theano.tensor as T
from theano import config
import theano
from blocks.algorithms import (GradientDescent, Adam, Momentum,
                               CompositeRule, StepClipping)
from blocks.extensions import FinishAfter, Printing, ProgressBar
from blocks.bricks.cost import CategoricalCrossEntropy, MisclassificationRate
from blocks.extensions.monitoring import (TrainingDataMonitoring,
                                          DataStreamMonitoring)
from blocks.bricks import Rectifier, Softmax, MLP
from blocks.main_loop import MainLoop
from blocks.model import Model
from utils import SaveLog, SaveParams, Glorot, visualize_attention, LRDecay
from blocks.initialization import Constant
from blocks.graph import ComputationGraph, apply_noise
from LSTM_attention_model import LSTMAttention
from blocks.monitoring import aggregation
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from visualize import analyze
floatX = theano.config.floatX
logger = logging.getLogger('main')


def setup_model(configs):
    tensor5 = theano.tensor.TensorType(config.floatX, (False,) * 5)
    # shape: T x B x C x X x Y
    input_ = tensor5('features')
    tensor3 = theano.tensor.TensorType(config.floatX, (False,) * 3)
    locs = tensor3('locs')
    # shape: B x Classes
    target = T.ivector('targets')

    model = LSTMAttention(
        configs,
        weights_init=Glorot(),
        biases_init=Constant(0))
    model.initialize()

    (h, c, location, scale, alpha, patch, downn_sampled_input,
        conved_part_1, conved_part_2, pre_lstm) = model.apply(input_, locs)

    model.location = location
    model.scale = scale
    model.alpha = location
    model.patch = patch

    classifier = MLP(
        [Rectifier(), Softmax()],
        configs['classifier_dims'],
        weights_init=Glorot(),
        biases_init=Constant(0))
    classifier.initialize()

    probabilities = classifier.apply(h[-1])
    cost = CategoricalCrossEntropy().apply(target, probabilities)
    cost.name = 'CE'
    error_rate = MisclassificationRate().apply(target, probabilities)
    error_rate.name = 'ER'
    model.cost = cost
    model.error_rate = error_rate
    model.probabilities = probabilities

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
        print "TESTING THE MODEL: CHECK THE INPUT SIZE!"
        cg = ComputationGraph(model.cost)
        f = theano.function(cg.inputs, [model.cost],
                            on_unused_input='ignore',
                            allow_input_downcast=True)
        data = configs['get_streams'](configs[
            'batch_size'])[0].get_epoch_iterator().next()
        f(data[1], data[0], data[2])

        print "Test passed! ;)"

    model.monitorings = [cost, error_rate]

    return model


def train(model, configs):
    get_streams = configs['get_streams']
    save_path = configs['save_path']
    num_epochs = configs['num_epochs']
    batch_size = configs['batch_size']
    lrs = configs['lrs']
    until_which_epoch = configs['until_which_epoch']
    grad_clipping = configs['grad_clipping']
    monitorings = model.monitorings

    # Training
    if configs['weight_noise'] > 0:
        cg = ComputationGraph(model.cost)
        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        cg = apply_noise(cg, weights, configs['weight_noise'])
        model.cost = cg.outputs[0].copy(name='CE')

    if configs['l2_reg'] > 0:
        cg = ComputationGraph(model.cost)
        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        new_cost = model.cost + configs['l2_reg'] * sum([
            (weight ** 2).sum() for weight in weights])
        model.cost = new_cost.copy(name='CE')

    blocks_model = Model(model.cost)
    all_params = blocks_model.parameters
    print "Number of found parameters:" + str(len(all_params))
    print all_params

    default_lr = np.float32(configs['lrs'][0])
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
        name = param.tag.annotations[0].name + "." + param.name
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
                       blocks_model, save_path,
                       after_epoch=True),
            SaveLog(after_epoch=True),
            ProgressBar(),
            LRDecay(lr_var, lrs, until_which_epoch,
                    after_epoch=True),
            Printing(after_epoch=True)])
    main_loop.run()


def evaluate(model, load_path, configs):
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

        inps = ComputationGraph(model.error_rate).inputs
        eval_function = theano.function(
            inps, [model.error_rate, model.probabilities])
        _, vds = configs['get_streams'](100)
        data = vds.get_epoch_iterator().next()
        print "Valid_ER: " + str(
            eval_function(data[0], data[2], data[1])[0])
        return eval_function


if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO)

        configs = {}
        # from datasets import get_cmv_v2_len10_streams
        # from datasets import get_cmv_v1_streams
        from datasets import get_bmnist_streams
        configs['get_streams'] = get_bmnist_streams
        configs['save_path'] = 'results/Test_'
        configs['num_epochs'] = 600
        configs['batch_size'] = 100
        configs['lrs'] = [1e-4, 1e-5, 1e-6]
        configs['until_which_epoch'] = [150, 400, configs['num_epochs']]
        configs['grad_clipping'] = 2
        configs['weight_noise'] = 0.0
        configs['conv_layers'] = [                          # 1  x 28  x 28
            ['conv_1', (20, 1, 5, 5), (2, 2), None],        # 20 x 16 x 16
            ['conv_2', (50, 20, 5, 5), (2, 2), None],       # 50 x 10 x 10
            ['conv_3', (80, 50, 3, 3), (2, 2), None]]       # 80 x 6 x 6
        configs['num_layers_first_half_of_conv'] = 0
        configs['fc_layers'] = [['fc', (2880, 128), 'relu']]
        configs['lstm_dim'] = 128
        configs['attention_mlp_hidden_dims'] = [128]
        configs['cropper_input_shape'] = (100, 100)
        configs['patch_shape'] = (28, 28)
        configs['num_channels'] = 1
        configs['classifier_dims'] = [configs['lstm_dim'], 64, 10]
        configs['load_pretrained'] = False
        configs['test_model'] = True
        configs['l2_reg'] = 0.001
        timestr = time.strftime("%Y_%m_%d_at_%H_%M")
        save_path = configs['save_path'] + timestr
        configs['save_path'] = save_path
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
            eval_function = evaluate(model, 'results/BMNIST_Learn_2016_02_25_at_23_50/', configs)
            analyze('results/BMNIST_Learn_2016_02_25_at_23_50/')
            visualize_attention(model, configs, eval_function)
        else:
            # evaluate(model, 'results/CMV_Hard_len10_2016_02_22_at_21_00/')
            train(model, configs)
