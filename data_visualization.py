import random
from matplotlib import pyplot as plt
from utils.save_utils import load_pickle


def main():
    video_paths = load_pickle('sbj_video_paths.pkl')
    subjects = list(video_paths.keys())
    emotions = list(video_paths[subjects[0]].keys())

    file_sufix = '_video_landmarks.pkl'
    file_name = f'{random.choice(subjects)}/{random.choice(emotions)}' + file_sufix
    print('loading ' + file_name)
    video_landmarks = load_pickle(file_name)
    video = video_landmarks[0]             # 1st video
    frame = video[0]
    x_lm = []
    y_lm = []
    z_lm = []
    for lm in frame:
        x_lm.append(lm[0]*16.0)
        y_lm.append(lm[1]*9.0)
        z_lm.append(lm[2]*16.0)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_lm, y_lm, z_lm)
    plt.show()
    

if __name__ == '__main__':
    main()
