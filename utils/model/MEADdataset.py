import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from utils.files.save import load_numpy, load_torch


class MEADDataset(Dataset):
    def __init__(self, partition: str, config: dict, feature: str = 'melspec', use_rescaled: bool = False, use_norm: bool = False):

        self.partition = partition
        self.consecutive_seqs = config['training']['consecutive_seqs']
        self.data_root = config['files'][partition]['root']
        self.landmarks_dir = 'landmarks'
        self.feature_dir = feature
        self.csv_file = config['files'][partition]['csv']
        self.csv_data = pd.read_csv(self.csv_file)
        self.mini_file = config['files'][partition]['mini']
        self.csv_mini = pd.read_csv(self.mini_file)
        self.subjects = config['files'][partition]['subjects']
        self.sbj_to_idx = {self.subjects[idx]: idx for idx in range(len(self.subjects))}
        self.emotions = config['emotions']
        self.e_to_idx = {self.emotions[idx]: idx for idx in range(len(self.emotions))}
        self.sample_rate = config['audio']['sample_rate']
        self.use_rescaled = use_rescaled
        self.use_norm = use_norm

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if index + self.consecutive_seqs - 1 >= len(self.csv_data):
            index = len(self.csv_data) - self.consecutive_seqs

        sbj, emo, feature_paths, lmks_paths, base_lmks_path = self._get_element_paths(index)
        seq_feature = torch.stack(self._get_element_data(feature_paths, 'numpy'))
        seq_targets = torch.stack(self._get_element_data(lmks_paths, 'numpy'))
        base_target = load_numpy(base_lmks_path)
        base_target = torch.from_numpy(base_target).type(torch.float32)
        base_target = base_target.repeat(self.consecutive_seqs, 1, 1)
        sbj_idx = [self.sbj_to_idx[sbj]] * self.consecutive_seqs
        sbj_idx = torch.Tensor(sbj_idx).type(torch.int64)
        emotion_idx = [self.e_to_idx[emo]] * self.consecutive_seqs
        emotion_idx = torch.Tensor(emotion_idx).type(torch.int64)

        return sbj_idx, emotion_idx, seq_feature, seq_targets, base_target

    def get_sequence(self, index: int):
        feature_container, lmks_container, sbj, e, lv, fname = self._get_inference_item_path(index)

        short_dir_feature = feature_container.replace('processed_data/', '')
        short_dir_lmks = lmks_container.replace('processed_data/', '')

        feature_paths = [os.path.join(short_dir_feature, file) for file in os.listdir(feature_container)]
        lmks_paths = [os.path.join(short_dir_lmks, file) for file in os.listdir(lmks_container)]

        feature_list = [torch.from_numpy(load_numpy(file)).type(torch.float32) for file in feature_paths]
        lmks_list = [torch.from_numpy(load_numpy(file)).type(torch.float32) for file in lmks_paths]

        suffix = self._get_file_suffix('lmks')
        base_target = load_numpy(os.path.join(self.partition, sbj, f'{sbj}{suffix}.npy'))
        base_target = torch.from_numpy(base_target).type(torch.float32)

        sbj_idx = [self.sbj_to_idx[sbj]] * len(feature_list)
        sbj_idx = torch.Tensor(sbj_idx).type(torch.int64)
        emotion_idx = [self.e_to_idx[e]] * len(feature_list)
        emotion_idx = torch.Tensor(emotion_idx).type(torch.int64)
        feature = torch.stack(feature_list)
        target = torch.stack(lmks_list)
        base_target = base_target.repeat(len(feature_list), 1, 1)

        return sbj_idx, emotion_idx, feature, target, base_target, sbj, e, lv, fname


    def _get_inference_item_path(self, index: int) -> tuple[str, str, str, str, str, str]:
        sbj, e, lv, audio, _ = self.csv_mini.iloc[index]
        fname = audio.split('.')[0]
        container_dir = os.path.join(self.data_root, sbj, e, f'level_{lv}')
        lmks_container = os.path.join(container_dir, self.landmarks_dir, fname)
        feature_container = os.path.join(container_dir, self.feature_dir, fname)
        return feature_container, lmks_container, sbj, e, f'level_{lv}', fname
    
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

    def _get_element_paths(self, index: int) -> tuple[str, str, str, str, str]:
        sbj, e, lv, audio, _, lmks, _ = self.csv_data.iloc[index]

        for i in range(1, self.consecutive_seqs):
            audio2 = self.csv_data['audio'].values[index + i]
            if audio != audio2:
                index += i - self.consecutive_seqs
                break

        lmks_list = []
        sbj, e, lv, audio, _, lmks, _ = self.csv_data.iloc[index]
        lmks_list.append(lmks)

        for i in range(1, self.consecutive_seqs):
            current_lmks = self.csv_data['landmarks'].values[index + i]
            lmks_list.append(current_lmks)

        audio = audio.split('.')[0]
        container_dir = os.path.join(self.data_root, sbj, e, f'level_{lv}')

        suffix = self._get_file_suffix('lmks')
        base_lmks_file = os.path.join(self.data_root, sbj, f'{sbj}{suffix}.npy')
        base_lmks_file = base_lmks_file.replace('processed_data/', '')

        target_lmks_files = self._get_sequence_paths('lmks', lmks_list, container_dir, audio)
        feature_files = self._get_sequence_paths('feature', lmks_list, container_dir, audio)

        return sbj, e, feature_files, target_lmks_files, base_lmks_file

    def _get_file_suffix(self, file_type: str = 'lmks'):
        if file_type == 'lmks':
            suffix = ''
            if self.use_rescaled:
                suffix += 'rs'
            if self.use_norm:
                suffix += 'n'
            return suffix
        else:
            return f'_{self.sample_rate}'
    
    def _get_sequence_paths(self, element_type: str, elements: list[str], container_dir: str, audio: str):
        if element_type == 'lmks':
            element_dir = self.landmarks_dir
        else:
            element_dir = self.feature_dir
        suffix = self._get_file_suffix(element_type)

        file_paths = []
        for e in elements:
            e = e.split('.')[0]
            file = os.path.join(container_dir, element_dir, audio, f'{e}{suffix}.npy')
            file = file.replace('processed_data/', '')
            file_paths.append(file)
        return file_paths
