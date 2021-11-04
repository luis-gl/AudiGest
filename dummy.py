import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch.optim

from config_creator import get_config
from utils.model.losses import *
from utils.model.SequenceRegressor import SequenceRegressor


def set_seed(seed: int, make_deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if make_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_last_epoch() -> list[int]:
    checkpoints_dir = os.path.join('processed_data', 'training')
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        print(f'No checkpoint detected.')
        return 0

    checkpoints = [int(file.split('.')[0].replace('AG_', ''))
                   for file in os.listdir(checkpoints_dir)
                   if os.path.isfile(os.path.join(checkpoints_dir, file))]

    if len(checkpoints) < 1:
        print(f'No checkpoint detected.')
        return []

    return sorted(checkpoints)


def plot_lr(epochs, lrs, train_loss, val_loss):
    x_values = range(1, len(train_loss) + 1)
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Learning Rate Every 5 Epochs')
    ax.plot(epochs, lrs, '-o')

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Loss per Epoch')
    ax.plot(x_values, train_loss, '-o', label='train')
    ax.plot(x_values, val_loss, '-o', label='val')
    ax.legend(loc='lower right')

    plt.show()


def main():
    config = get_config()

    lr = config['training']['learning_rate']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    model = SequenceRegressor(config, device, feature_type='melspec')

    checkpoints = get_last_epoch()
    lrs = []
    epochs = []

    train_loss = []
    val_loss = []
    for saved_epoch in checkpoints:
        _, scheduler_st, train_loss, val_loss = model.load(saved_epoch)

        if scheduler_st is not None:
            # print(scheduler_st)
            epoch = scheduler_st['last_epoch']
            last_lr = scheduler_st['_last_lr']
            print(f'epoch {epoch}: {last_lr}\t{train_loss[epoch-1]}')
            epochs.append(epoch)
            lrs.append(last_lr)

    plot_lr(epochs, lrs, train_loss[200:], val_loss[200:])


if __name__ == '__main__':
    main()
