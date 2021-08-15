import os
import pickle


def load_dictionary(file_name):
    data_dir = 'processed_data'
    file_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise ValueError('{} does not exists'.format(file_path))

    with open(file_path, 'rb') as handle:
        return pickle.load(handle)

def save_dictionary(data_dict, file_name):
    data_dir = 'processed_data'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)

    file_path = os.path.join(data_dir, file_name)

    with open(file_path, 'wb') as handle:
        pickle.dump(data_dict, handle)