import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from utils.files.save import load_numpy, load_torch


class MEADDataset(Dataset):
    def __init__(self, train: bool, config: dict):

        self.train = train
        self.data_root = config['files']['train_root'] if train else config['files']['test_root']
        self.audio_dir = 'clean_audio' if config['audio']['use_clean'] else 'audio'
        self.landmarks_dir = 'landmarks'
        self.melspec_dir = 'melspec'
        self.mfcc_dir = 'mfcc'
        self.csv_file = config['files']['train_csv'] if train else config['files']['test_csv']
        self.csv_data = pd.read_csv(self.csv_file)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        melspec_path, lmks_path, mfcc_path = self._get_element_path(index)
        melspec = load_torch(melspec_path)
        target = load_numpy(lmks_path)
        target = torch.from_numpy(target)
        mfcc = load_torch(mfcc_path)

        return melspec, mfcc, target

    def _get_element_path(self, index: int) -> tuple[str, str, str]:
        sbj, e, lv, audio, melspec, lmks, mfcc = self.csv_data.iloc[index]
        audio = audio.split('.')[0]
        container_dir = os.path.join(self.data_root, sbj, e, f'level_{lv}')
        melspec_file = os.path.join(container_dir, self.melspec_dir, melspec)
        melspec_file = melspec_file.replace('processed_data/', '')
        landmarks_file = os.path.join(container_dir, self.landmarks_dir, audio, lmks)
        landmarks_file = landmarks_file.replace('processed_data/', '')
        mfcc_file = os.path.join(container_dir, self.mfcc_dir, audio, mfcc)
        mfcc_file = mfcc_file.replace('processed_data/', '')
        return melspec_file, landmarks_file, mfcc_file
