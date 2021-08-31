from matplotlib import pyplot as plt
from utils.save_utils import load_numpy


def main():
    video = load_numpy('test.npy')
    frame = video[0]
    x_lm = []
    y_lm = []
    z_lm = []
    for lm in frame:
        x_lm.append(lm[0])
        y_lm.append(lm[1])
        z_lm.append(lm[2])

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_lm, y_lm, z_lm)
    plt.show()
    

if __name__ == '__main__':
    main()
