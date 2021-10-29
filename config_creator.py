import os
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def set_config(config: dict):
    with open('config.yml', 'w') as yml_file:
        yaml.dump(config, yml_file, Dumper=Dumper)


def create_default_config():
    config = {
        'files': {
            'raw_data_root': 'MEAD',
            'data_root': 'processed_data',
            'subject_paths': 'processed_data/sbj_data_paths.pkl',
            'train': {
                'root': 'processed_data/train',
                'csv': 'processed_data/train_dataset.csv',
                'mini': 'processed_data/train_mini.csv',
                'subjects': ['M005', 'M007', 'M011', 'M012', 'M022', 'M023', 'M024', 'M025', 'M026', 'W014', 'W016']
            },
            'val': {
                'root': 'processed_data/val',
                'csv': 'processed_data/val_dataset.csv',
                'mini': 'processed_data/val_mini.csv',
                'subjects': ['M009', 'M013', 'W011', 'W015']
            },
            'test': {
                'root': 'processed_data/test',
                'csv': 'processed_data/test_dataset.csv',
                'mini': 'processed_data/test_mini.csv',
                'subjects': ['M003', 'M019', 'W009', 'W019']
            },
            'face': 'processed_data/face.obj'
        },
        'audio': {
            'use_clean': True,
            'sample_rate': 44100,
            'max_samples': 44100,
            'sample_interval': 1 / 30,
            'window_len': 0.025,
            'n_mfcc': 20,
            'fps': 30
        },
        'emotions': ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised'],
        'model': {
            'lstm': {
                'hidden_dim': 128,
                'num_layers': 1,
            },
            'fc': {
                'in_feat': 576,
                'hidden_1': 256,
                'hidden_2': 128,
                'hidden_3': 256,
                'hidden_4': 512,
                'drop': 0.5
            },
            'vertex_num': 468,
            'velocity_weight': 10.0
        },
        'training': {
            'batch_size': 16,
            'learning_rate': 4e-6,
            'epochs': 40,
            'decay_rate': 1.0,
            'consecutive_seqs': 2
        }
    }

    set_config(config)

    return config


def get_config():
    if not os.path.exists('config.yml'):
        print('Creating default config')
        return create_default_config()

    with open('config.yml', 'r') as yml_file:
        return yaml.load(yml_file, Loader=Loader)


if __name__ == '__main__':
    create_default_config()
