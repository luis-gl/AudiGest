import numpy as np
import os
import pandas as pd

from tqdm import tqdm
from utils.files.save import load_numpy, save_numpy


def convert_to_txt(seq, fname):
    txt = f'{len(seq)}\n'
    for frame in seq:
        txt += '\n'
        for lm in frame:
            txt += f'{lm[0]},{lm[1]},{lm[2]}\n'
    
    with open(fname, 'w') as f:
        f.write(txt)
        f.close()


def main():
    for phase in ['train', 'val', 'test']:
        phase_data = pd.read_csv(f'processed_data/{phase}_subjects.csv')
        loop = tqdm(range(len(phase_data)))
        for i in loop:
            sbj, e, lv, _, lmks = phase_data.iloc[i]
            lmks_path = os.path.join(phase, sbj, e, f'level_{lv}', 'landmarks', lmks)
            seq_lmks = load_numpy(lmks_path)
            centers = seq_lmks.mean(axis=1)
            magnitudes = np.sqrt(np.einsum('ij,ij->i', centers, centers))
            min_mgntd_idx = magnitudes.argmin()
            min_center = centers[min_mgntd_idx]
            seq_lmks[:] -= min_center
            save_numpy(seq_lmks, file_name=lmks_path)


if __name__ == '__main__':
    main()