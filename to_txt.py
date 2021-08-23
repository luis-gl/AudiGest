from utils.save_utils import load_pickle


def main():
    file_path = 'M003/angry_video_landmarks.pkl'
    videos = load_pickle(file_path)
    video = videos[0]
    txt = f'{len(video)}\n'
    for frame in video:
        txt += '\n'
        for lm in frame:
            txt += f'{lm[0]},{lm[1]},{lm[2]}\n'

    with open('video.txt', 'w') as f:
        f.write(txt)
        f.close()


if __name__ == '__main__':
    main()