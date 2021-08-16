from utils.save_utils import load_pickle, save_pickle


def main():
    face_landmarks = load_pickle('sbj_face_landmarks.pkl')
    file_sufix = '_video_landmarks.pkl'
    c = 0
    for sbj in face_landmarks:
        for emotion in face_landmarks[sbj]:
            save_pickle(data=face_landmarks[sbj][emotion],
                            file_name=emotion + file_sufix,
                            dir_path=sbj)
        c += 1
        if c > 3:
            break


if __name__ == '__main__':
    main()
