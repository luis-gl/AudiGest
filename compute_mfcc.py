import math
import os
import pandas as pd

from config_creator import get_config
from threading import Thread
from utils.files.save import load_numpy, save_numpy, save_torch
from utils.audio import *


def generate_subject_data(min_max: tuple[int, int], csv_data: pd.DataFrame, phase: str, phase_path: str,
                          config: dict, melspec_transform, mfcc_transform):
    min_v, max_v = min_max
    for i in range(min_v, max_v):
        sbj, e, lv, audio, lmks = csv_data.iloc[i]
        sbj_path = os.path.join(phase_path, sbj, e, f'level_{lv}')
        audio_path = os.path.join(sbj_path, 'audio', audio)
        lmks_path = os.path.join(phase, sbj, e, f'level_{lv}', 'landmarks', lmks)
        melspec_path = os.path.join(phase, sbj, e, f'level_{lv}', 'melspec')

        signal = get_signal_mono(audio_path, config=config)
        melspec = get_sliced_melspectrogram(signal, ms_transform=melspec_transform, max_samples=config['max_samples'])
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
    print(min_max)


def main():
    config = get_config()

    data_root = 'processed_data'
    phase = 'train'
    audio_cfg = config['audio']
    n_fft = int(audio_cfg['window_len'] * audio_cfg['sample_rate'])
    hop_len = int(audio_cfg['sample_interval'] * audio_cfg['sample_rate'])
    melspec_t = get_melspec_transform(audio_cfg['sample_rate'], n_fft, hop_len)
    mfcc_t = get_mfcc_transform(audio_cfg['sample_rate'], audio_cfg['n_mfcc'], n_fft=n_fft, hop_len=hop_len)

    phase_path = os.path.join(data_root, phase)
    csv_path = os.path.join(data_root, f'{phase}_data.csv')
    csv_data = pd.read_csv(csv_path)

    n = len(csv_data)
    print('total:', n)

    num_threads = 12
    increment = math.ceil(n / num_threads)
    jobs = []
    for idx in range(num_threads):
        min_v, max_v = increment * idx, increment * (idx + 1)
        if max_v > n:
            max_v = n
        args = ((min_v, max_v), csv_data, phase, phase_path, audio_cfg, melspec_t, mfcc_t)
        task = Thread(target=generate_subject_data, args=args)
        jobs.append(task)
        task.start()

    for job in jobs:
        job.join()


if __name__ == '__main__':
    main()
