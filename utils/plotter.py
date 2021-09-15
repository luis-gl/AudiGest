import matplotlib.pyplot as plt


def plot_loss(model_dic, model_name=None, save=False, test=False):
    train_loss_history = model_dic['train_history']
    val_loss_history = model_dic['val_history']
    x_values = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(7, 5))
    if model_name is None:
        plt.title('Loss')
    else:
        plt.title(model_name + ' Loss')
    plt.plot(x_values, train_loss_history, '-o', label='train')
    plt.plot(x_values, val_loss_history, '-o', label='val')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    if save:
        if test:
            fname = f'./plots/{model_name}_losses_test.png'
        else:
            fname = f'./plots/{model_name}_losses.png'
        plt.savefig(fname)
        return
    plt.show()
