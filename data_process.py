import os
from face_detection import FaceDetector
from threading import Thread
from utils.file_utils.convert_to_wav import convert_to_wav
from utils.file_utils.data_utils import get_data
from utils.file_utils.save_utils import save_numpy


def process_subject_data(subject: str, data_dict: dict, file_type: str, detector: FaceDetector = None):
    for emotion in data_dict[subject]:
        for level in data_dict[subject][emotion]:
            audio_paths = data_dict[subject][emotion][level]['audio']
            videos_paths = data_dict[subject][emotion][level]['video']

            output_dir = f'{subject}/{emotion}/{level}/'

            for i in range(len(audio_paths)):
                if file_type == 'audio':
                    convert_to_wav(file=audio_paths[i],
                                   out_folder='processed_data/' + output_dir + 'audio',
                                   clear_meta_data=True,
                                   out_folder_metadata='processed_data/' + output_dir + 'clean_audio')
                elif file_type == 'video':
                    file_name = videos_paths[i]
                    video_landmarks = detector.get_face_landmarks(video_path=file_name)
                    file_name = os.path.basename(file_name)
                    file_name = file_name.split('.')[0]
                    save_numpy(np_data=video_landmarks, file_name=f'{file_name}.npy',
                               dir_path=output_dir + 'landmarks')


def main():
    data_dict = get_data()
    detector = FaceDetector(confidence=0.8)
    jobs = []
    for subject in data_dict:
        task = Thread(target=process_subject_data, args=(subject, data_dict, 'audio'))
        jobs.append(task)
        task.start()

    for subject in data_dict:
        process_subject_data(subject, data_dict, 'video', detector)

    for job in jobs:
        job.join()


if __name__ == '__main__':
    main()
