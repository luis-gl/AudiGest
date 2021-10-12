import os
import pandas as pd
from tqdm import tqdm

from config_creator import get_config
from utils.files.save import load_numpy, save_numpy, save_torch
from utils.audio import *


def generate_subject_data(csv_data: pd.DataFrame, phase: str, phase_path: str, config: dict,
                          n_fft: int, hop_len: int, mfcc_transform):
    for i in tqdm(range(len(csv_data))):
        sbj, e, lv, audio, lmks = csv_data.iloc[i]
        sbj_path = os.path.join(phase_path, sbj, e, f'level_{lv}')
        audio_path = os.path.join(sbj_path, 'audio', audio)
        lmks_path = os.path.join(phase, sbj, e, f'level_{lv}', 'landmarks', lmks)
        melspec_path = os.path.join(phase, sbj, e, f'level_{lv}', 'melspec')

        signal = get_signal_mono(audio_path, config=config)
        melspec = get_sliced_melspectrogram(signal, config=config, n_fft=n_fft, hop_len=hop_len)
        melspec_fname = audio.split('.')[0]
        save_torch(melspec, file_name=f'{melspec_fname}.pt', dir_path=melspec_path)
        mfcc_list = process_framed_mfcc(signal, config=config, mfcc_transform=mfcc_transform)
        video_landmarks = load_numpy(lmks_path)

        n_mfcc_list = len(mfcc_list)
        n_video_lmks = len(video_landmarks)
        if n_mfcc_list < n_video_lmks:
            video_landmarks = video_landmarks[:n_mfcc_list - n_video_lmks]
        elif n_mfcc_list > n_video_lmks:
            mfcc_list = mfcc_list[:n_video_lmks - n_mfcc_list]

        assert len(mfcc_list) > 0 and len(video_landmarks) > 0
        assert len(mfcc_list) == len(video_landmarks)

        mfcc_path = os.path.join(phase, sbj, e, f'level_{lv}', 'mfcc', lmks)
        mfcc_path = mfcc_path.split('.')[0]
        lmks_path = lmks_path.split('.')[0]

        for idx, mfcc in enumerate(mfcc_list):
            save_torch(mfcc, file_name=f'{(idx + 1):03d}.pt', dir_path=mfcc_path)
            save_numpy(video_landmarks[idx], file_name=f'{(idx + 1):03d}.npy', dir_path=lmks_path)


def main():
    config = get_config()

    data_root = 'processed_data'
    phases = ['train', 'val', 'test']
    audio_cfg = config['audio']
    n_fft = int(audio_cfg['window_len'] * audio_cfg['sample_rate'])
    hop_len = int(audio_cfg['sample_interval'] * audio_cfg['sample_rate'])
    mfcc_t = get_mfcc_transform(audio_cfg['sample_rate'], audio_cfg['n_mfcc'], n_fft=n_fft, hop_len=hop_len)

    for phase in phases:
        phase_path = os.path.join(data_root, phase)
        csv_path = os.path.join(data_root, f'{phase}_subjects.csv')
        csv_data = pd.read_csv(csv_path)

        generate_subject_data(csv_data, phase, phase_path, config=audio_cfg, n_fft=n_fft,
                            hop_len=hop_len, mfcc_transform=mfcc_t)


if __name__ == '__main__':
    main()
