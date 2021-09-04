import os
import pickle
import numpy as np


def load_numpy(file_name: str = '') -> np.ndarray:
    data_dir = 'processed_data'
    file_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise ValueError('{} does not exists'.format(file_path))

    return np.load(file_path)


def save_numpy(np_data: np.ndarray, file_name: str, dir_path: str = None):
    data_dir = 'processed_data'
    if dir_path is not None:
        data_dir = os.path.join(data_dir, dir_path)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_path = os.path.join(data_dir, file_name)
    np.save(file_path, np_data)


def load_pickle(file_name='') -> object:
    """
        Load pickle serialized data from processed_data directory.

        Args:
            file_name: String containing the file path from 'processed_data' directory.

        Raises:
            ValueError: If the file does not exists.

        Returns:
            The content of the pickle serialized file.
    """
    data_dir = 'processed_data'
    file_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        raise ValueError('{} does not exists'.format(file_path))

    with open(file_path, 'rb') as handle:
        return pickle.load(handle)


def save_pickle(data, file_name, dir_path=None):
    """
        Save content to pickle serialized file in 'processed_data' directory.

        Args:
            data: Object containing the data to save in file.
            file_name: String containing the name for the file (only name, not directory).
            dir_path: String containing the directory for the file if the user want to save
            the file inside a folder.
    """
    data_dir = 'processed_data'
    if dir_path is not None:
        data_dir = os.path.join(data_dir, dir_path)

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    file_path = os.path.join(data_dir, file_name)
    with open(file_path, 'wb') as handle:
        pickle.dump(data, handle)