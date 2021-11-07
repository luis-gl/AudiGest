import numpy as np
import os
import pandas as pd

from tqdm import tqdm
from utils.files.save import load_numpy, save_numpy


def convert_to_txt(seq, fname):
    txt = f'{len(seq)}\n'
    for i, frame in enumerate(seq):
        txt += '\n'
        for j, lm in enumerate(frame):
            txt += f'{lm[0]},{lm[1]},{lm[2]}'
            if i < len(seq) - 1 or j < len(frame) - 1:
                txt += '\n'

    file_path = os.path.join('processed_data/visualization', fname)
    with open(file_path, 'w') as f:
        f.write(txt)
        f.close()


def transform_landmark_files(phase, lmks_path, lmks_fname):
    scaled_lmks_file = os.path.join(lmks_path, f'{lmks_fname}rs.npy')
    norm_sc_lmks_file = os.path.join(lmks_path, f'{lmks_fname}rsn.npy')

    rs_seq_lmks = load_numpy(scaled_lmks_file)
    convert_to_txt(rs_seq_lmks, f'{lmks_fname}rs_{phase}.txt')
    norm_rs_seq_lmks = load_numpy(norm_sc_lmks_file)
    convert_to_txt(norm_rs_seq_lmks, f'{lmks_fname}rsn_{phase}.txt')


def normalize_sequence(seq: np.ndarray, rs_seq: np.ndarray):
    seq_lmks = seq.copy()
    rs_seq_lmks = rs_seq.copy()
    centers = seq_lmks.mean(axis=1)
    rs_centers = rs_seq_lmks.mean(axis=1)
    for idx in range(seq_lmks.shape[0]):
        seq_lmks[idx, :] -= centers[idx]
        rs_seq_lmks[idx, :] -= rs_centers[idx]
    return seq_lmks, rs_seq_lmks

def center_sequence(seq: np.ndarray):
    seq_lmks = seq.copy()
    centers = seq_lmks.mean(axis=1)
    for idx in range(seq_lmks.shape[0]):
        seq_lmks[idx, :] -= centers[idx]
    return seq_lmks


def main():
    for phase in ['train', 'val']:
        phase_data = pd.read_csv(f'processed_data/{phase}_dataset.csv')
        subjects = phase_data['subject'].unique()
        loop = tqdm(subjects, total=len(subjects))
        for sbj in loop:
            lmks_path = os.path.join(phase, sbj)
            lmks_file = os.path.join(lmks_path, f'{sbj}.npy')
            seq_lmks = load_numpy(lmks_file)
            seq_lmks = center_sequence(seq_lmks)
            save_numpy(seq_lmks, file_name=f'{sbj}c.npy', dir_path=lmks_path)


if __name__ == '__main__':
    main()
