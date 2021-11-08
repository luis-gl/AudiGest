import numpy as np
import matplotlib.pyplot as  plt
import torch
import torch.nn.functional as F

from landmark_normalization import convert_to_txt
from random import randint
from utils.audio import AudioFeatureExtractor
from utils.files.save import load_numpy
from utils.model.SequenceRegressor import SequenceRegressor
from utils.model.MEADdataset import MEADDataset

from config_creator import get_config
from training import get_last_epoch
# from utils.rendering.model_render import ModelRender


def mse(predicted: torch.tensor, target: torch.tensor) -> float:
    return torch.square(predicted - target).mean().item()


def graph_face(faces: np.ndarray, rows: int, cols: int, start_idx: int):
    n = rows * cols
    fig = plt.figure()
    for i in range(start_idx, start_idx + n):
        face = faces[i].T
        ax = fig.add_subplot(rows, cols, i - start_idx + 1, projection='3d')
        ax.scatter(face[0], face[1], face[2])
    plt.show()


def make_inference(model: SequenceRegressor, device: torch.device, dataset: MEADDataset = None,
                    extractor: AudioFeatureExtractor = None) -> torch.tensor:
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        if dataset is not None:
            _, emotion, feature, _, template = dataset[0]
        elif extractor is not None:
            audio_path = 'processed_data/audios/EMODB_03a07Fb.wav'
            template_path = 'val/M013/M013c.npy'

            emotion = torch.Tensor([randint(0,7)]).type(torch.int64)

            _, mfccs, melspec = extractor.get_melspec_and_mfccs(audio_path=audio_path)
            feature = mfccs
            feature = torch.from_numpy(feature).type(torch.float32)

            template = load_numpy(template_path)
            template = torch.from_numpy(template).type(torch.float32)
            template = template.repeat(feature.shape[1], 1, 1)

        # print('emotion:', emotion.shape)
        # print('feature:', feature.shape)
        # print('template:', template.shape)
        # print('-' * 10)

        subject = F.one_hot(torch.Tensor([1]).type(torch.int64), 6)

        emotion = F.one_hot(emotion, 8)

        feature = feature.unsqueeze(dim=0)
        feature = feature.permute(0, 2, 1)

        template = template.unsqueeze(dim=0)

        subject = subject.to(device)
        emotion = emotion.to(device)
        feature = feature.to(device)
        template = template.to(device)

        reconstructed = model(feature, emotion, subject, template)
        return reconstructed


def main():
    config = get_config()

    test_data = MEADDataset(partition='val', config=config, use_centered=True)

    
    feature_extractor = AudioFeatureExtractor(config['audio'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    model = SequenceRegressor(config, device)
    last_epoch = get_last_epoch()
    model.load(last_epoch)

    # renderer = ModelRender(config=config, dataset=test_data)
    # renderer.render_sequences(model, device, 'output/videos')

    reconstructed = make_inference(model, device, dataset=test_data)
    reconstructed = reconstructed.squeeze(dim=0)
    reconstructed = reconstructed.cpu().numpy()
    # pth = 'val/M013/angry/level_1/landmarks/001c.npy'
    # reconstructed = load_numpy(pth)
    print('reconstructed:', reconstructed.shape)

    # convert_to_txt(reconstructed, '001_inf2.txt')
    graph_face(reconstructed, 1, 2, 0)

if __name__ == '__main__':
    main()