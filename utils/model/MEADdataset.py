import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from utils.files.save import load_numpy, load_torch


class MEADDataset(Dataset):
    def __init__(self, partition: str, config: dict, use_centered: bool = False):

        self.partition = partition
        self.data_root = config['files'][partition]['root']
        self.landmarks_dir = 'landmarks'
        self.feature_dir = config['model']['feature']
        self.csv_file = config['files'][partition]['csv']
        self.csv_data = pd.read_csv(self.csv_file)
        self.subjects = config['files']['train']['subjects']
        self.sbj_to_idx = {self.subjects[idx]: idx for idx in range(len(self.subjects))}
        self.emotions = config['emotions']
        self.e_to_idx = {self.emotions[idx]: idx for idx in range(len(self.emotions))}
        self.sample_rate = config['audio']['sample_rate']
        self.use_centered = use_centered

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index: int):
        sbj, emo, feature_path, lmks_path, template_path = self._get_element_paths(index)
        feature_seq = load_numpy(feature_path)
        feature_seq = torch.from_numpy(feature_seq).type(torch.float32)

        lmks_seq = load_numpy(lmks_path)
        lmks_seq = torch.from_numpy(lmks_seq).type(torch.float32)

        sbj_template = load_numpy(template_path)
        sbj_template = torch.from_numpy(sbj_template).type(torch.float32)
        sbj_template = sbj_template.repeat(lmks_seq.shape[0], 1, 1)

        sbj_idx = [0]
        sbj_idx = torch.Tensor(sbj_idx).type(torch.int64)

        emotion_idx = [self.e_to_idx[emo]]
        emotion_idx = torch.Tensor(emotion_idx).type(torch.int64)

        return sbj_idx, emotion_idx, feature_seq, lmks_seq, sbj_template

    def _get_element_paths(self, index: int):
        sbj, e, lv, audio = self.csv_data.iloc[index]
        audio = f'{audio:03d}'
        container_dir = os.path.join(self.data_root, sbj, e, f'level_{lv}')
        container_dir = container_dir.replace('processed_data/', '')

        lmks_suffix = self._get_file_suffix('lmks')
        feature_suffix = self._get_file_suffix('feature')

        base_lmks_file = os.path.join(self.data_root, sbj, f'{sbj}{lmks_suffix}.npy')
        base_lmks_file = base_lmks_file.replace('processed_data/', '')

        target_lmks_file = os.path.join(container_dir, self.landmarks_dir, f'{audio}{lmks_suffix}.npy')
        feature_dir = os.path.join(container_dir, self.feature_dir)
        feature_file = os.path.join(feature_dir, f'{audio}{feature_suffix}.npy')

        return sbj, e, feature_file, target_lmks_file, base_lmks_file

    def _get_file_suffix(self, file_type: str = 'lmks'):
        if file_type == 'lmks':
            suffix = ''
            if self.use_centered:
                suffix += 'c'
            return suffix
        else:
            return f'_{self.sample_rate}'
    
    def _get_sequence_paths(self, element_type: str, elements: 'list[str]', container_dir: str, audio: str):
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
