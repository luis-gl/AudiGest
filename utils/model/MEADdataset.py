import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from utils.files.save import load_numpy, load_torch


class MEADDataset(Dataset):
    def __init__(self, partition: str, config: dict):

        self.partition = partition
        self.consecutive_seqs = config['training']['consecutive_seqs']
        self.data_root = config['files'][partition]['root']
        self.audio_dir = 'clean_audio' if config['audio']['use_clean'] else 'audio'
        self.landmarks_dir = 'landmarks'
        self.melspec_dir = 'melspec'
        self.mfcc_dir = 'mfcc'
        self.csv_file = config['files'][partition]['csv']
        self.csv_data = pd.read_csv(self.csv_file)
        self.mini_file = config['files'][partition]['mini']
        self.csv_mini = pd.read_csv(self.mini_file)

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if index + self.consecutive_seqs - 1 >= len(self.csv_data):
            index = len(self.csv_data) - self.consecutive_seqs
        
        melspec_path, mfcc_paths, lmks_paths, base_lmks_path = self._get_element_paths(index)
        melspec = load_torch(melspec_path)
        melspec = melspec.repeat(len(mfcc_paths), 1, 1)
        seq_mfcc = torch.stack(self._get_element_data(mfcc_paths, 'torch'))
        seq_targets = torch.stack(self._get_element_data(lmks_paths, 'numpy'))
        base_target = load_numpy(base_lmks_path)
        base_target = torch.from_numpy(base_target).type(torch.float32)
        base_target = base_target.repeat(len(mfcc_paths), 1, 1)

        return melspec, seq_mfcc, seq_targets, base_target

    def get_sequence(self, index: int):
        melspec_file, mfcc_container, lmks_container, sbj, e, lv, fname = self._get_inference_item_path(index)
        melspec = load_torch(melspec_file)

        short_dir_mfcc = mfcc_container.replace('processed_data/', '')
        short_dir_lmks = lmks_container.replace('processed_data/', '')

        mfcc_paths = [os.path.join(short_dir_mfcc, file) for file in os.listdir(mfcc_container)]
        lmks_paths = [os.path.join(short_dir_lmks, file) for file in os.listdir(lmks_container)]

        mfcc_list = [load_torch(file) for file in mfcc_paths]
        lmks_list = [torch.from_numpy(load_numpy(file)).type(torch.float32) for file in lmks_paths]

        base_target = load_numpy(os.path.join(self.partition, sbj, 'base.npy'))
        base_target = torch.from_numpy(base_target).type(torch.float32)

        melspec = melspec.repeat(len(mfcc_list), 1, 1)
        mfcc = torch.stack(mfcc_list)
        target = torch.stack(lmks_list)
        base_target = base_target.repeat(len(mfcc_list), 1, 1)

        return melspec, mfcc, target, base_target, sbj, e, lv, fname


    def _get_inference_item_path(self, index: int) -> tuple[str, str, str]:
        sbj, e, lv, audio, _ = self.csv_mini.iloc[index]
        fname = audio.split('.')[0]
        container_dir = os.path.join(self.data_root, sbj, e, f'level_{lv}')
        melspec_file = os.path.join(container_dir, self.melspec_dir, f'{fname}.pt')
        melspec_file = melspec_file.replace('processed_data/', '')
        lmks_container = os.path.join(container_dir, self.landmarks_dir, fname)
        mfcc_container = os.path.join(container_dir, self.mfcc_dir, fname)
        return melspec_file, mfcc_container, lmks_container, sbj, e, f'level_{lv}', fname
    
    def _get_element_data(self, element_paths, load_type='torch'):
        data_list = []
        for e in element_paths:
            if load_type == 'torch':
                data = load_torch(e)
            else:
                data = load_numpy(e)
                data = torch.from_numpy(data).type(torch.float32)
            data_list.append(data)
        return data_list

    def _get_element_paths(self, index: int) -> tuple[str, str, str, str]:
        sbj, e, lv, audio, melspec, lmks, mfcc = self.csv_data.iloc[index]

        for i in range(1, self.consecutive_seqs):
            _, _, _, audio2, _, _, _ = self.csv_data.iloc[index + i]
            if audio != audio2:
                index += i - self.consecutive_seqs
                break

        lmks_list = []
        mfcc_list = []
        sbj, e, lv, audio, melspec, lmks, mfcc = self.csv_data.iloc[index]
        lmks_list.append(lmks)
        mfcc_list.append(mfcc)

        for i in range(1, self.consecutive_seqs):
            _, _, _, _, _, current_lmks, current_mfcc = self.csv_data.iloc[index + i]
            lmks_list.append(current_lmks)
            mfcc_list.append(current_mfcc)

        audio = audio.split('.')[0]
        container_dir = os.path.join(self.data_root, sbj, e, f'level_{lv}')

        melspec_file = os.path.join(container_dir, self.melspec_dir, melspec)
        melspec_file = melspec_file.replace('processed_data/', '')

        base_lmks_file = os.path.join(self.data_root, sbj, 'base.npy')
        base_lmks_file = base_lmks_file.replace('processed_data/', '')

        target_lmks_files = self._get_sequence_paths('lmks', lmks_list, container_dir, audio)
        mfcc_files = self._get_sequence_paths('mfcc', mfcc_list, container_dir, audio)

        return melspec_file, mfcc_files, target_lmks_files, base_lmks_file
    
    def _get_sequence_paths(self, element_type, elements, container_dir, audio):
        if element_type == 'lmks':
            element_dir = self.landmarks_dir
        else:
            element_dir = self.mfcc_dir

        file_paths = []
        for e in elements:
            file = os.path.join(container_dir, element_dir, audio, e)
            file = file.replace('processed_data/', '')
            file_paths.append(file)
        return file_paths
