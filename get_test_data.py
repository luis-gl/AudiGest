import os
from config_creator import get_config
from face_detection import FaceDetector
from utils.files.convert_to_wav import convert_to_wav
from utils.files.save import save_numpy


def main():
    subjects = ['M011', 'W014']
    root = 'MEAD/TEST'
    selected_audio = 'level_1/007.m4a'
    selected_video = 'level_1/007.mp4'

    config = get_config()
    emotions = config['emotions']
    data_root = config['files']['data_root']
    data_root = os.path.join(data_root, 'test')
    detector = FaceDetector(confidence=0.8)

    for subject in subjects:
        subject_root = os.path.join(root, subject)
        for emotion in emotions:
            audio_file = os.path.join(subject_root, 'audio', emotion, selected_audio)
            video_file = os.path.join(subject_root, 'video/front', emotion, selected_video)

            output_dir = os.path.join(subject, emotion, 'level_1')
            audio_out_folder = os.path.join(data_root, output_dir, 'audio')
            audio_out_folder_metadata = os.path.join(data_root, output_dir, 'clean_audio')
            video_dir_path = os.path.join('test', output_dir, 'landmarks')

            # print('audio file:', audio_file)
            # print('audio out folder:', audio_out_folder)
            # print('audio meta:', audio_out_folder_metadata)
            convert_to_wav(audio_file, audio_out_folder, True, audio_out_folder_metadata)
            # print('video file:', video_file)
            video_lmks = detector.get_face_landmarks(video_path=video_file)
            # print('video lmks:', video_lmks.shape)
            fname = os.path.basename(video_file).split('.')[0]
            # print('video fname:', fname)
            # print('video dir path:', video_dir_path)
            # print('-'*10)
            save_numpy(video_lmks, file_name=f'{fname}.npy', dir_path=video_dir_path)


if __name__ == '__main__':
    main()