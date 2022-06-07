import os
import pandas as pd
from tqdm import tqdm

from config_creator import get_config
from landmark_normalization import center_sequence
from utils.files.save import load_numpy, save_numpy
from utils.audio import AudioFeatureExtractor


def generate_subject_data(csv_data: pd.DataFrame,
                          phase: str,
                          phase_path: str,
                          feature_extractor: AudioFeatureExtractor):
    sample_rate = feature_extractor.sample_rate
    for i in tqdm(range(len(csv_data))):
        sbj, e, lv, audio = csv_data.iloc[i]
        audio = f'{audio:03d}'
        sbj_path = os.path.join(phase_path, sbj, e, f'level_{lv}')
        audio_path = os.path.join(sbj_path, 'audio', f'{audio}.wav')
        lmks_path = os.path.join(phase, sbj, e, f'level_{lv}', 'landmarks')
        lmks_file = os.path.join(lmks_path, f'{audio}.npy')
        melspec_path = os.path.join(phase, sbj, e, f'level_{lv}', 'melspec')
        mfcc_path = os.path.join(phase, sbj, e, f'level_{lv}', 'mfcc')

        _, mfccs, melspec = feature_extractor.get_melspec_and_mfccs(audio_path)        

        video_landmarks = load_numpy(lmks_file)

        mfcc_frames = mfccs.shape[1]
        melspec_frames = melspec.shape[1]
        if mfcc_frames < melspec_frames:
            melspec = melspec[:, :mfcc_frames - melspec_frames]
        elif mfcc_frames > melspec_frames:
            mfccs = mfccs[:, :melspec_frames - mfcc_frames]

        mfcc_frames = mfccs.shape[1]
        melspec_frames = melspec.shape[1]
        assert mfcc_frames > 0 and melspec_frames > 0
        assert mfcc_frames == melspec_frames


        mfcc_frames = mfccs.shape[1]
        video_frames = video_landmarks.shape[0]
        if mfcc_frames < video_frames:
            video_landmarks = video_landmarks[:mfcc_frames - video_frames]
        elif mfcc_frames > video_frames:
            mfccs = mfccs[:, :video_frames - mfcc_frames]
            melspec = melspec[:, :video_frames - mfcc_frames]

        mfcc_frames = mfccs.shape[1]
        video_frames = video_landmarks.shape[0]
        melspec_frames = melspec.shape[1]
        assert mfcc_frames > 0 and video_frames > 0 and melspec_frames > 0
        assert mfcc_frames == video_frames and melspec_frames == video_frames

        melspec_save_path = os.path.join(melspec_path, f'{sample_rate}')
        mfcc_save_path = os.path.join(mfcc_path, f'{sample_rate}')
        lmks_save_path = os.path.join(lmks_path, f'{sample_rate}')

        save_numpy(melspec, file_name=f'{audio}_{sample_rate}.npy', dir_path=melspec_save_path)
        save_numpy(mfccs, file_name=f'{audio}_{sample_rate}.npy', dir_path=mfcc_save_path)

        save_numpy(video_landmarks, file_name=f'{audio}_{sample_rate}.npy', dir_path=lmks_save_path)
        centered_landmarks = center_sequence(video_landmarks)
        save_numpy(centered_landmarks, file_name=f'{audio}_{sample_rate}c.npy', dir_path=lmks_save_path)


def main():
    config = get_config()

    phases = ['train', 'val']
    audio_cfg = config['audio']
    feature_extractor = AudioFeatureExtractor(audio_cfg)

    for phase in phases:
        phase_path = config['files'][phase]['root']
        csv_path = config['files'][phase]['csv']
        csv_data = pd.read_csv(csv_path)

        generate_subject_data(csv_data, phase, phase_path, feature_extractor)


if __name__ == '__main__':
    main()
