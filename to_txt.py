import numpy as np


def main():
    video = np.load('test.npy')
    txt = f'{len(video)}\n'
    for frame in video:
        txt += '\n'
        for lm in frame:
            txt += f'{lm[0]},{lm[1]},{lm[2]}\n'

    with open('video2.txt', 'w') as f:
        f.write(txt)
        f.close()


if __name__ == '__main__':
    main()