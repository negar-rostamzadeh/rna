import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(path, to_be_plotted):
    results = {}
    log = open(path, 'r').readlines()
    for line in log:
        colon_index = line.find(":")
        enter_index = line.find("\n")
        if colon_index != -1:
            key = line[:colon_index]
            value = line[colon_index + 1: enter_index]
            if key in to_be_plotted:
                try:
                    value = float(value)
                    values = results.get(key)
                    if values is None:
                        results[key] = [value]
                    else:
                        results[key] = results[key] + [value]

                except ValueError:
                    print ""
    return results


def pimp(path=None, xaxis='Epochs', yaxis='Cross Entropy', title=None):
    plt.legend(fontsize=14)
    plt.xlabel(r'\textbf{' + xaxis + '}')
    plt.ylabel(r'\textbf{' + yaxis + '}')
    plt.grid()
    plt.title(r'\textbf{' + title + '}')
    plt.ylim([0, 1.5])
    if path is not None:
        plt.savefig(path)
    else:
        plt.show()


def plot(x, xlabel):
    x_steps = range(len(x))
    plt.plot(x_steps, x, lw=1, label=xlabel)


def analyze(path, save_path='plot.png'):
    to_be_plotted = [
        'train_lstmattention.w_1_mlp_norm',
        'train_lstmattention.W_pre_lstm_norm',
        'train_lstmattention.W_state_norm',
        'train_linear_0.W_norm',
        'train_lstmattention.initial_state_norm',

        'train_lstmattention.w_1_mlp_grad_norm',
        'train_lstmattention.W_pre_lstm_grad_norm',
        'train_lstmattention.W_state_grad_norm',
        'train_linear_0.W_grad_norm',
        'train_lstmattention.initial_state_grad_norm',

        'train_ER',
        'valid_ER']

    plt.figure(figsize=(20, 10))
    results = parse_log(path + 'log.txt',
                        to_be_plotted)
    plt.subplot(521)
    for key in to_be_plotted[:5]:
        values = results[key]
        plot(values, key)

    plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.4, 1))
    plt.grid()

    plt.subplot(523)
    for key in to_be_plotted[5:10]:
        values = results[key]
        plot(values, key)

    plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.4, 1))
    plt.grid()

    plt.subplot(525)
    for key in to_be_plotted[10:]:
        values = results[key]
        plot(values, key)

    plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1.4, 1))
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=250)

    print 'Best train_ER: ' + str(np.min(results['train_ER']))
    print 'Best valid_ER: ' + str(np.min(results['valid_ER']))

# analyze('results/BMNIST_CNN_Learn_2016_02_26_at_12_32/', 'plot.png')
