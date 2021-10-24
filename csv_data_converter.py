import os

from config_creator import get_config


def get_csv_elements(parent_dir: str, sample_rate: int):
    csv_line = ''
    for content in os.listdir(parent_dir):
        content_path = os.path.join(parent_dir, content)
        if not os.path.isdir(content_path):
            continue

        if not content.startswith('level'):
            csv_line += get_csv_elements(content_path, sample_rate)
            continue

        audio_path = os.path.join(content_path, 'audio')
        melspec_path = os.path.join(content_path, 'melspec')
        lmks_path = os.path.join(content_path, 'landmarks')
        mfcc_path = os.path.join(content_path, 'mfcc')
        csv_elements = content_path.replace('processed_data\\', '')
        csv_elements = csv_elements.replace('train\\', '')
        csv_elements = csv_elements.replace('test\\', '')
        csv_elements = csv_elements.replace('\\', ',')
        csv_elements = csv_elements.replace('level_', '')
        for name in os.listdir(audio_path):
            audio_file = os.path.join(audio_path, name)
            audio_exists = os.path.exists(audio_file) and os.path.isfile(audio_file)
            name = name.split('.')[0]
            melspec_file = os.path.join(melspec_path, f'{name}_{sample_rate}.npy')
            melspec_exists = os.path.exists(melspec_file) and os.path.isfile(melspec_file)
            if not audio_exists or not melspec_exists:
                print(f'{audio_file} or {melspec_file} does not exists.')
                continue
            audio_lmks_path = os.path.join(lmks_path, name)
            audio_mfcc_path = os.path.join(mfcc_path, name)
            audio_csv = f'{csv_elements},{name}.wav,{name}.npy,'
            if not os.path.exists(audio_lmks_path):
                print(f'{audio_lmks_path} does not exists.')
                continue
            if not os.path.exists(audio_mfcc_path):
                print(f'{audio_mfcc_path} does not exists.')
                continue
            for frame in os.listdir(audio_lmks_path):
                frame = frame.split('.')[0]
                if 'n' in frame or 'rs' in frame:
                    continue
                lmks_file = os.path.join(audio_lmks_path, f'{frame}.npy')
                mfcc_file = os.path.join(audio_mfcc_path, f'{frame}_{sample_rate}.npy')
                exist_lmks = os.path.exists(lmks_file) and os.path.isfile(lmks_file)
                exist_mfcc = os.path.exists(mfcc_file) and os.path.isfile(mfcc_file)
                if not exist_lmks or not exist_mfcc:
                    print(f'{lmks_file} or {mfcc_file} does not exists')
                    continue
                csv_line += audio_csv + f'{frame}.npy,{frame}.npy\n'

    return csv_line


def main():
    config = get_config()
    root = 'processed_data'
    col_names = 'subject,emotion,level,audio,melspec,landmarks,mfccs\n'
    for state in ['train', 'val', 'test']:
        csv = get_csv_elements(os.path.join(root, state), config['audio']['sample_rate'])
        with open(f'processed_data/{state}_dataset.csv', 'w') as file:
            file.write(col_names + csv)


if __name__ == '__main__':
    main()
