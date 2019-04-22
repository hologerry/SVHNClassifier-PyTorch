import argparse
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logdir', default='./logs', help='directory to read logs')


def _visualize(path_to_log_dir):
    losses = np.load(os.path.join(path_to_log_dir, 'losses.npy'))
    accuracies = np.load(os.path.join(path_to_log_dir, 'accuracies.npy'))

    num_steps_to_check = 1000
    x_loss = [i for i in range(len(losses))]
    x_acc = [i*num_steps_to_check for i in range(len(accuracies))]
    loss_data = pd.DataFrame({"step": x_loss, "loss": losses})
    acc_data = pd.DataFrame({"step": x_acc, "accuracy": accuracies})
    fig, axs = plt.subplots(ncols=2)
    sns.lineplot(x='step', y='loss', data=loss_data, ax=axs[0])
    sns.lineplot(x='step', y='accuracy', data=acc_data, ax=axs[1])
    plt.tight_layout()
    plt.show()


def main(args):
    path_to_log_dir = args.logdir
    _visualize(path_to_log_dir)


if __name__ == '__main__':
    main(parser.parse_args())
