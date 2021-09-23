import os
import pandas as pd
import shutil


def check_dir(root: str, dir_path: str) -> str:
    complete_path = os.path.join(root, dir_path)
    if not os.path.exists(complete_path):
        os.makedirs(complete_path)
    
    return complete_path


def copy_file(common_dir: str, orig_root: str, dest_root: str,
            fname: str = None, dir_name: str = None):

    dest_path = check_dir(dest_root, common_dir)
    orig_path = os.path.join(orig_root, common_dir)

    if fname is not None:
        orig_path = os.path.join(orig_path, fname)
        shutil.copy2(orig_path, dest_path)
    elif dir_name is not None:
        orig_path = os.path.join(orig_path, dir_name)
        dest_path = os.path.join(dest_path, dir_name)
        shutil.copytree(orig_path, dest_path, dirs_exist_ok=True)


def main():
    data_root = 'processed_data'
    dest_root = os.path.join(data_root, 'mini')
    phases = ['train', 'test']

    for phase in phases:
        csv_file = os.path.join(data_root, f'{phase}_mini.csv')
        data = pd.read_csv(csv_file)
        n = len(data)
        for i in range(n):
            sbj, e, lv, audio, _ = data.iloc[i]
            name = audio.split('.')[0]
            
            common_path = os.path.join(phase, sbj, e, f'level_{lv}')

            audio_dir = os.path.join(common_path, 'clean_audio')
            lmks_dir = os.path.join(common_path, 'landmarks')
            melspec_dir = os.path.join(common_path, 'melspec')
            mfcc_dir = os.path.join(common_path, 'mfcc')

            copy_file(audio_dir, data_root, dest_root, fname=f'{name}.wav')
            copy_file(melspec_dir, data_root, dest_root, fname=f'{name}.pt')
            copy_file(lmks_dir, data_root, dest_root, dir_name=name)
            copy_file(mfcc_dir, data_root, dest_root, dir_name=name)


if __name__ == '__main__':
    main()