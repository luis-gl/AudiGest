import os
import pandas as pd
from tqdm import tqdm

from config_creator import get_config
from face_detection import rescale_landmark_sequence
from landmark_normalization import normalize_sequence
from utils.files.save import load_numpy, save_numpy
from utils.audio import AudioFeatureExtractor


def generate_subject_data(csv_data: pd.DataFrame,
                          phase: str,
                          phase_path: str,
                          feature_extractor: AudioFeatureExtractor):
    sample_rate = feature_extractor.sample_rate
    for i in tqdm(range(len(csv_data))):
        sbj, e, lv, audio, lmks = csv_data.iloc[i]
        sbj_path = os.path.join(phase_path, sbj, e, f'level_{lv}')
        audio_path = os.path.join(sbj_path, 'audio', audio)
        lmks_path = os.path.join(phase, sbj, e, f'level_{lv}', 'landmarks')
        lmks_file = os.path.join(lmks_path, lmks)
        melspec_path = os.path.join(phase, sbj, e, f'level_{lv}', 'melspec')
        mfcc_path = os.path.join(phase, sbj, e, f'level_{lv}', 'mfcc')
        audio_fname = audio.split('.')[0]

        short_melspec, mfccs, melspec = feature_extractor.get_melspec_and_mfccs(audio_path)
        save_numpy(short_melspec, file_name=f'{audio_fname}_{sample_rate}.npy', dir_path=melspec_path)
        save_numpy(melspec, file_name=f'{audio_fname}full_{sample_rate}.npy', dir_path=melspec_path)
        save_numpy(mfccs, file_name=f'{audio_fname}_{sample_rate}.npy', dir_path=mfcc_path)

        framed_mfccs = feature_extractor.process_framed_feature(mfccs, 'mfcc')
        framed_melspec = feature_extractor.process_framed_feature(melspec, 'melspec')
        video_landmarks = load_numpy(lmks_file)

        mfcc_frames = framed_mfccs.shape[0]
        video_frames = video_landmarks.shape[0]
        if mfcc_frames < video_frames:
            video_landmarks = video_landmarks[:mfcc_frames - video_frames]
        elif mfcc_frames > video_frames:
            framed_mfccs = framed_mfccs[:video_frames - mfcc_frames]
            framed_melspec = framed_melspec[:video_frames - mfcc_frames]

        mfcc_frames = framed_mfccs.shape[0]
        video_frames = video_landmarks.shape[0]
        melspec_frames = framed_melspec.shape[0]
        assert mfcc_frames > 0 and video_frames > 0 and melspec_frames > 0
        assert mfcc_frames == video_frames and melspec_frames == video_frames

        save_numpy(framed_mfccs, file_name=f'{audio_fname}_{sample_rate}f.npy', dir_path=mfcc_path)
        save_numpy(framed_melspec, file_name=f'{audio_fname}full_{sample_rate}f.npy', dir_path=melspec_path)
        save_numpy(video_landmarks, file_name=f'{audio_fname}.npy', dir_path=lmks_path)
        rescaled_landmarks = rescale_landmark_sequence(video_landmarks)
        save_numpy(rescaled_landmarks, file_name=f'{audio_fname}rs.npy', dir_path=lmks_path)
        norm_video_landmarks, norm_rs_landmarks = normalize_sequence(video_landmarks, rescaled_landmarks)
        save_numpy(norm_video_landmarks, file_name=f'{audio_fname}n.npy', dir_path=lmks_path)
        save_numpy(norm_rs_landmarks, file_name=f'{audio_fname}rsn.npy', dir_path=lmks_path)

        framed_mfcc_path = os.path.join(mfcc_path, audio_fname)
        framed_melspec_path = os.path.join(melspec_path, audio_fname)
        framed_lmks_path = os.path.join(lmks_path, audio_fname)

        for idx in range(video_frames):
            save_numpy(framed_mfccs[idx], file_name=f'{(idx + 1):03d}_{sample_rate}.npy', dir_path=framed_mfcc_path)
            save_numpy(framed_melspec[idx], file_name=f'{(idx + 1):03d}_{sample_rate}.npy', dir_path=framed_melspec_path)
            save_numpy(video_landmarks[idx], file_name=f'{(idx + 1):03d}.npy', dir_path=framed_lmks_path)
            save_numpy(rescaled_landmarks[idx], file_name=f'{(idx + 1):03d}rs.npy', dir_path=framed_lmks_path)
            save_numpy(norm_video_landmarks[idx], file_name=f'{(idx + 1):03d}n.npy', dir_path=framed_lmks_path)
            save_numpy(norm_rs_landmarks[idx], file_name=f'{(idx + 1):03d}rsn.npy', dir_path=framed_lmks_path)


def main():
    config = get_config()

    data_root = config['files']['data_root']
    phases = ['train', 'val', 'test']
    audio_cfg = config['audio']
    feature_extractor = AudioFeatureExtractor(audio_cfg)

    for phase in phases:
        phase_path = config['files'][phase]['root']
        csv_path = os.path.join(data_root, f'{phase}_subjects.csv')
        csv_data = pd.read_csv(csv_path)

        generate_subject_data(csv_data, phase, phase_path, feature_extractor)


if __name__ == '__main__':
    main()
