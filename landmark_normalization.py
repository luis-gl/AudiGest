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


def main():
    for phase in ['train', 'val', 'test']:
        phase_data = pd.read_csv(f'processed_data/{phase}_subjects.csv')
        loop = tqdm(range(len(phase_data)))
        for i in loop:
            sbj, e, lv, _, lmks = phase_data.iloc[i]
            lmks = lmks.split('.')[0]
            lmks_path = os.path.join(phase, sbj, e, f'level_{lv}', 'landmarks')
            # transform_landmark_files(phase, lmks_path, lmks)
            lmks_file = os.path.join(lmks_path, f'{lmks}.npy')
            scaled_lmks_file = os.path.join(lmks_path, f'{lmks}rs.npy')
            seq_lmks = load_numpy(lmks_file)
            rs_seq_lmks = load_numpy(scaled_lmks_file)
            centers = seq_lmks.mean(axis=1)
            rs_centers = rs_seq_lmks.mean(axis=1)
            for idx in range(seq_lmks.shape[0]):
                seq_lmks[idx, :] -= centers[idx]
                rs_seq_lmks[idx, :] -= rs_centers[idx]
            save_numpy(seq_lmks, file_name=f'{lmks}n.npy', dir_path=lmks_path)
            save_numpy(rs_seq_lmks, file_name=f'{lmks}rsn.npy', dir_path=lmks_path)


if __name__ == '__main__':
    main()
