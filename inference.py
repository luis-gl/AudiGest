import numpy as np
import matplotlib.pyplot as  plt
import torch

from utils.model.AudiGest import AudiGest
from utils.model.MEADdataset import MEADDataset

from config_creator import get_config
from training import get_last_epoch
# from utils.rendering.model_render import ModelRender


def mse(predicted: torch.tensor, target: torch.tensor) -> float:
    return torch.square(predicted - target).mean().item()


def graph_face(faces: np.ndarray, rows: int, cols: int):
    n = rows * cols
    fig = plt.figure()
    for i in range(n):
        face = faces[i].T
        ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
        ax.scatter(face[0], face[1], face[2])
    plt.show()


def make_inference(model: AudiGest, device: torch.device, melspec: torch.Tensor, mfcc: torch.Tensor,
                    base_target: torch.Tensor) -> torch.tensor:
    model = model.to(device)
    model.eval()
    hidden = None

    with torch.no_grad():
        melspec = melspec.unsqueeze(1)
        melspec = melspec.to(device)

        mfcc = mfcc.permute(0, 2, 1)
        mfcc = mfcc.to(device)

        base_target = base_target.to(device)

        reconstructed, _ = model(melspec, mfcc, base_target, hidden)
        return reconstructed


def main():
    config = get_config()

    # train_data = MEADDataset(train=True, config=config)
    test_data = MEADDataset(partition='test', config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    model = AudiGest(config)
    last_epoch = get_last_epoch()
    model.load(last_epoch)

    # renderer = ModelRender(config=config, dataset=test_data)
    # renderer.render_sequences(model, device, 'output/videos')

    melspec, mfcc, target, base_target, _, _, _, _ = test_data.get_sequence(15)
    
    print('melspec:', melspec.shape)
    print('mfcc:', mfcc.shape)
    print('base:', base_target.shape)
    print('target:', target.shape)

    reconstructed = make_inference(model, device, melspec, mfcc, base_target)
    reconstructed = reconstructed.cpu()
    print('reconstructed:', reconstructed.shape)

    graph_face(reconstructed, 1, 2)
    # print('mse:', mse(reconstructed, target))
    
    # npy_face = reconstructed.numpy()
    # save_numpy(npy_face, 'test.npy')


if __name__ == '__main__':
    main()