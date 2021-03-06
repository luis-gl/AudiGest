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
                'subjects': ['M003','M009','M019','W009','W011','W019']
            },
            'val': {
                'root': 'processed_data/val',
                'csv': 'processed_data/val_dataset.csv',
                'subjects': ['M013','W015']
            },
            'test': {
                'root': 'processed_data/test',
                'csv': 'processed_data/test_dataset.csv',
                'subjects': ['M011', 'W014']
            },
            'face': 'processed_data/face.obj'
        },
        'audio': {
            'sample_rate': 44100,
            'max_samples': 44100,
            'sample_interval': 1 / 30,
            'window_len': 0.025,
            'n_mfcc': 20,
            'fps': 30
        },
        'emotions': ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised'],
        'model': {
            'hidden_dim': 128,
            'num_layers': 1,
            'vertex_num': 468,
            'feature': 'mfcc',
            'use_condition': True,
            'velocity_weight': 10.0
        },
        'training': {
            'batch_size': 1,
            'learning_rate': 1e-6,
            'epochs': 600,
            'decay_rate': 1.0
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
