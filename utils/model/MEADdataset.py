import os
import pandas as pd
import torch
from torch.nn.modules import container

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
        self.mini_file = config['files']['train_mini'] if train else config['files']['test_mini']
        self.csv_mini = pd.read_csv(self.mini_file)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        melspec_path, lmks_path, mfcc_path = self._get_element_path(index)
        melspec = load_torch(melspec_path)
        target = load_numpy(lmks_path)
        target = torch.from_numpy(target).type(torch.float32)
        mfcc = load_torch(mfcc_path)

        return melspec, mfcc, target

    def get_sequence(self, index: int):
        melspec_file, mfcc_container, lmks_container, sbj, e, lv, fname = self._get_inference_item_path(index)
        melspec = load_torch(melspec_file)

        short_dir_mfcc = mfcc_container.replace('processed_data/', '')
        short_dir_lmks = lmks_container.replace('processed_data/', '')

        mfcc_paths = [os.path.join(short_dir_mfcc, file) for file in os.listdir(mfcc_container)]
        lmks_paths = [os.path.join(short_dir_lmks, file) for file in os.listdir(lmks_container)]

        mfcc_list = [load_torch(file) for file in mfcc_paths]
        lmks_list = [torch.from_numpy(load_numpy(file)) for file in lmks_paths]

        melspec = melspec.repeat(len(mfcc_list), 1, 1)
        mfcc = torch.stack(mfcc_list)
        target = torch.stack(lmks_list)

        return melspec, mfcc, target, sbj, e, lv, fname


    def _get_inference_item_path(self, index: int) -> tuple[str, str, str]:
        sbj, e, lv, audio, _ = self.csv_mini.iloc[index]
        fname = audio.split('.')[0]
        container_dir = os.path.join(self.data_root, sbj, e, f'level_{lv}')
        melspec_file = os.path.join(container_dir, self.melspec_dir, f'{fname}.pt')
        melspec_file = melspec_file.replace('processed_data/', '')
        lmks_container = os.path.join(container_dir, self.landmarks_dir, fname)
        mfcc_container = os.path.join(container_dir, self.mfcc_dir, fname)
        return melspec_file, mfcc_container, lmks_container, sbj, e, f'level_{lv}', fname

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
