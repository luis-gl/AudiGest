import torch
from torch.utils.data import dataset

from utils.model.AudiGest import AudiGest
from utils.model.MEADdataset import MEADDataset

from config_creator import get_config
from training import get_last_epoch
from utils.rendering.model_render import ModelRender
from utils.files.save import save_numpy


def mse(predicted: torch.tensor, target: torch.tensor) -> float:
    return torch.square(predicted - target).mean().item()


def make_inference(model: AudiGest, device: torch.device, melspec: torch.Tensor, mfcc: torch.Tensor) -> torch.tensor:
    model = model.to(device)
    model.eval()
    hidden = None

    with torch.no_grad():
        melspec = melspec.unsqueeze(1)
        melspec = melspec.to(device)

        mfcc = mfcc.permute(0, 2, 1)
        mfcc = mfcc.to(device)

        reconstructed, _ = model(melspec, mfcc, hidden)
        return reconstructed


def main():
    config = get_config()

    # train_data = MEADDataset(train=True, config=config)
    test_data = MEADDataset(train=False, config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using {device}')

    model = AudiGest(config)
    last_epoch = get_last_epoch()
    model.load(last_epoch)

    renderer = ModelRender(config=config, dataset=test_data)
    renderer.render_sequences(model, device, 'output/videos')

    # melspec, mfcc, target, _, _, _ = test_data.get_sequence(0)
    # print('melspec:', melspec.shape)
    # print('mfcc:', mfcc.shape)
    # print('target:', target.shape)

    # reconstructed = make_inference(model, device, melspec, mfcc)
    # reconstructed = reconstructed.cpu()
    # print('reconstructed:', reconstructed.shape)
    # print('mse:', mse(reconstructed, target))
    
    # npy_face = reconstructed.numpy()
    # save_numpy(npy_face, 'test.npy')


if __name__ == '__main__':
    main()