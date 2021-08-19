import os
from face_detection import FaceDetector
from utils.save_utils import save_pickle, load_pickle


def get_valid_subjects_paths():
    data_path = 'MEAD/'
    subjects = [sbj for sbj in os.listdir(data_path) if not sbj.startswith('I')]
    common_path = '/video/front/'
    emotions = os.listdir(data_path + subjects[0] + common_path)
    emotions_set = set(emotions)

    for sbj in subjects:
        sbj_emotions = os.listdir(data_path + sbj + common_path)
        if set(sbj_emotions) != emotions_set:
            subjects.remove(sbj)

    neutral_lv_dir = '/level_1/'
    emotion_lv_dir = '/level_2/'
    subject_dict = {}
    for sbj in subjects:
        subject_dict[sbj] = {}
        for e in emotions:
            lv_path = neutral_lv_dir if e == 'neutral' else emotion_lv_dir
            video_path = data_path + sbj + common_path + e + lv_path
            videos = os.listdir(video_path)
            paths = [video_path + v for v in videos]
            subject_dict[sbj][e] = paths

    return subject_dict


def get_subject_video_data():
    sbj_video_paths_fname = 'sbj_video_paths.pkl'
    sbj_video_paths = None
    try:
        sbj_video_paths = load_pickle(sbj_video_paths_fname)
    except ValueError:
        print('Creating new subject video paths data.')
        sbj_video_paths = get_valid_subjects_paths()
        save_pickle(sbj_video_paths, sbj_video_paths_fname)
    else:
        print('Loading existing subject video paths data.')
    finally:
        return sbj_video_paths


def print_subject_dict(data_dict):
    for sbj in data_dict:
        print(f'{sbj}:')
        for e in data_dict[sbj]:
            print(f'\t{e}:')
            for v in data_dict[sbj][e]:
                print(f'\t\t{v}')


def main():
    video_data = get_subject_video_data()
    detector = FaceDetector()
    file_sufix = '_video_landmarks.pkl'
    for sbj in video_data:
        print(sbj)
        for emotion in video_data[sbj]:
            face_landmarks = []
            for video in video_data[sbj][emotion]:
                video_landmarks = detector.get_face_landmarks(video_path=video)
                face_landmarks.append(video_landmarks)
            save_pickle(data=face_landmarks, file_name=emotion + file_sufix,
                        dir_path=sbj)


if __name__ == '__main__':
    main()
