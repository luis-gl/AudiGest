from face_detection import FaceDetector
from threading import Thread
from utils.convert_to_wav import convert_to_wav
from utils.data_utils import get_data
from utils.save_utils import save_numpy


def process_subject_data(subject: str, data_dict: dict, face_detector: FaceDetector):
    for emotion in data_dict[subject]:
        for level in data_dict[subject][emotion]:
            audio_paths = data_dict[subject][emotion][level]['audio']
            videos_paths = data_dict[subject][emotion][level]['video']

            output_dir = f'{subject}/{emotion}/{level}/'

            for i in range(len(audio_paths)):
                convert_to_wav(audio_paths[i], out_folder='processed_data/' + output_dir + 'audio',
                               clear_meta_data=True, out_folder_metadata='processed_data/' + output_dir + 'clean_audio')

                video_landmarks = face_detector.get_face_landmarks(video_path=videos_paths[i])
                save_numpy(np_data=video_landmarks, file_name=f'{i:03}.npy',
                           dir_path=output_dir + 'landmarks')


def main():
    data_dict = get_data()
    detector = FaceDetector(confidence=0.8)
    jobs = []
    for subject in data_dict:
        print(subject)
        task = Thread(target=process_subject_data, args=(subject, data_dict, detector,))
        jobs.append(task)
        task.start()


if __name__ == '__main__':
    main()
