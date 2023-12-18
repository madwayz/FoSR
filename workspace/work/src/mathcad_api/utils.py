import numpy as np
from matplotlib.pyplot import show


def create_timeline(*args):
    """
    На вход подаются данные для создания цикла, обёрнутые в np.arrange()
    """

    return [dig for dig in np.arange(*args)]


def tolist(array):
    return np.ndarray.tolist(array)


def write_in_file(data, path, text):
    with open(path, 'w+') as f:
        f.write(data)

        print('{} был(-о/и) записан(-о/ы) в {}'.format(text, path))


def save_plot(plt, name, path, xCoords=None, yCoords=None, label=None, xlabel=None, ylabel=None):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if xCoords is not None and yCoords is not None:
        if label is not None:
            plt.plot(xCoords, yCoords, label=label)
        else:
            plt.plot(xCoords, yCoords)
    else:
        plt.plot(xCoords, yCoords)

    if label is not None:
        plt.legend(bbox_to_anchor=(1.01, 0.15), loc='right')
    plt.savefig(path + name)
    show()

    print('Сохраняю график в {}'.format(path + name))

