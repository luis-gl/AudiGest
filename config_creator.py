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
            'train_root': 'processed_data/train',
            'test_root': 'processed_data/test',
            'train_csv': 'processed_data/train_dataset.csv',
            'test_csv': 'processed_data/test_dataset.csv',
        },
        'audio': {
            'use_clean': False,
            'sample_rate': 16000,
            'max_samples': 16000,
            'sample_interval': 1 / 30,
            'window_len': 0.025,
            'n_mfcc': 20,
            'fps': 30
        },
        'model': {
            'cnn': {
                'kernel_1': 5
            },
            'lstm': {
                'input_dim': 20,
                'hidden_dim': 40,
                'num_layers': 2,
                'output_dim': 256
            },
            'fc': {
                'hidden_1': 512,
                'lmks_num': 468
            }
        },
        'training': {
            'batch_size': 64
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
    get_config()
