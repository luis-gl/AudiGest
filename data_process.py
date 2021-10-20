import os
from concurrent.futures import ThreadPoolExecutor
from config_creator import get_config
from face_detection import FaceDetector
from utils.files.convert_to_wav import convert_to_wav
from utils.files.data import get_data
from utils.files.save import save_numpy


def process_subject_data(phase: str, subject: str, data_dict: dict, data_root: str, file_type: str, detector: FaceDetector = None):
    sbj_dict = data_dict[subject]
    for emotion in sbj_dict:
        e_dict = sbj_dict[emotion]
        for level in e_dict:
            audio_paths = e_dict[level]['audio']
            videos_paths = e_dict[level]['video']

            output_dir = f'{subject}/{emotion}/{level}'
            audio_out_folder = os.path.join(data_root, output_dir, 'audio')
            audio_out_folder_metadata = os.path.join(data_root, output_dir, 'clean_audio')
            video_dir_path = os.path.join(phase, output_dir, 'landmarks')

            for i in range(len(audio_paths)):
                if file_type == 'audio':
                    convert_to_wav(file=audio_paths[i],
                                   out_folder=audio_out_folder,
                                   clear_meta_data=True,
                                   out_folder_metadata=audio_out_folder_metadata)
                elif file_type == 'video':
                    file_name = videos_paths[i]
                    video_landmarks = detector.get_face_landmarks(video_path=file_name)
                    file_name = os.path.basename(file_name)
                    file_name = file_name.split('.')[0]
                    save_numpy(np_data=video_landmarks, file_name=f'{file_name}.npy',
                               dir_path=video_dir_path)


def main():
    config = get_config()
    data_dict = get_data(config)
    detector = FaceDetector(confidence=0.8)

    with ThreadPoolExecutor(max_workers=5) as pool:
        for phase in data_dict:
            phase_dict = data_dict[phase]
            data_root = config['files'][phase]['root']
            for subject in phase_dict:
                pool.submit(process_subject_data, phase, subject, phase_dict, data_root, 'audio')

    for phase in data_dict:
        phase_dict = data_dict[phase]
        data_root = config['files'][phase]['root']
        for subject in phase_dict:
            process_subject_data(phase, subject, phase_dict, data_root, 'video', detector)


if __name__ == '__main__':
    main()
