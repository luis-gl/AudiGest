import numpy as np
import matplotlib.pyplot as  plt
import torch
import torch.nn.functional as F

from landmark_normalization import convert_to_txt
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


def make_inference(model: SequenceRegressor, device: torch.device, dataset: MEADDataset) -> torch.tensor:
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        _, emotion, feature, _, template = dataset[0]

        # print('emotion:', emotion.shape)
        # print('feature:', feature.shape)
        # print('template:', template.shape)
        # print('-' * 10)

        subject = F.one_hot(torch.Tensor([1]).type(torch.int64), 4)

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

    test_data = MEADDataset(partition='train', config=config, feature='melspec', use_rescaled=False, use_norm=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    model = SequenceRegressor(config, device, feature_type='melspec')
    last_epoch = get_last_epoch()
    model.load(last_epoch)

    # renderer = ModelRender(config=config, dataset=test_data)
    # renderer.render_sequences(model, device, 'output/videos')

    reconstructed = make_inference(model, device, test_data)
    reconstructed = reconstructed.squeeze(dim=0)
    reconstructed = reconstructed.cpu().numpy()
    print('reconstructed:', reconstructed.shape)

    convert_to_txt(reconstructed, '001_inf.txt')
    # graph_face(reconstructed, 1, 2, 0)
    # print('mse:', mse(reconstructed, target))
    
    # npy_face = reconstructed.numpy()
    # save_numpy(npy_face, 'test.npy')


if __name__ == '__main__':
    main()