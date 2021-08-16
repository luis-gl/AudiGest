import os
import pickle


def load_pickle(file_name):
    data_dir = 'processed_data'
    file_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise ValueError('{} does not exists'.format(file_path))

    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def save_pickle(data, file_name, dir_path=None):
    data_dir = 'processed_data'
    if dir_path is not None:
        data_dir = os.path.join(data_dir, dir_path)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_path = os.path.join(data_dir, file_name)
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle)