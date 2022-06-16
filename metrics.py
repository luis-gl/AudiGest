import os
import pandas as pd
import numpy as np

from config_creator import get_config
from utils.files.save import load_numpy
from inference import make_inference

def RMSE(predicted, original):
    return np.sqrt(np.mean((predicted - original)**2))

def main():
    config = get_config()
    data_path = config['files']['test']['root']
    sample_rate = config['audio']['sample_rate']
    csv_file = config['files']['test']['csv']
    csv_data = pd.read_csv(csv_file)

    for i in range(len(csv_data)):
        sbj, e, lv, audio = csv_data.iloc[i]
        sbj_path = os.path.join(data_path, sbj, e, f'level_{lv}')
        audio_path = os.path.join(sbj_path, 'audio', f'{audio:03d}.wav')
        face_path = os.path.join('test', sbj, f'{sbj}c.npy')
        original_path = os.path.join(sbj_path, 'landmarks', f'{sample_rate}', f'{audio:03d}_{sample_rate}c.npy')
        original_path = original_path.replace('processed_data/', '')
        original = load_numpy(original_path)
        predicted = make_inference(audio_path, e, face_path, '', '')
        rmse = RMSE(predicted, original)
        print(f'{sbj} - {e}: {rmse:.7f}')

if __name__ == '__main__':
    main()