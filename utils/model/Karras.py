import torch
import os
from torch import nn
from utils.files.save import load_torch, save_torch


class KarrasModel(nn.Module):
    def __init__(self, config: dict, input_type: str = 'melspec'):
        super(KarrasModel, self).__init__()

        self.config = config

        self.analysis = AudioAnalyzer(input_type=input_type)
        self.articulation = Articulation()

        self.output = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=68, out_features=config['model']['vertex_num']*3),
            nn.Tanh(),
        )

    def forward(self, feature: torch.Tensor, emotion: torch.Tensor, subject: torch.Tensor, subject_template):
        emotion = emotion.reshape(-1, emotion.shape[1], 1, 1)
        emotion = torch.tile(emotion, (1, 1, feature.shape[-2], feature.shape[-1]))

        feature = torch.cat((feature, emotion), dim=1)  # [B, 1, melspec/mfcc, 30] -> [B, 9, melspec/mfcc, 30]
        out_analysis = self.analysis(feature)      # [B, 9, melspec/mfcc, 30] -> [B, 64, 1, 30]

        subject = subject.reshape(-1, subject.shape[-1], 1, 1)
        subject_tiled = torch.tile(subject, (1, 1, 1, feature.shape[-1]))

        out_analysis = torch.cat((out_analysis, subject_tiled), dim=1) # [B, 64, 1, 30] -> [B, 68, 1, 30]
        out_articulation = self.articulation(out_analysis)    # [B, 68, 1, 30] -> [B, 64, 1, 1]

        out_articulation = torch.cat((out_articulation, subject), dim=1) # [B, 64, 1, 1] -> [B, 68, 1, 1]
        offsets = self.output(out_articulation)    # [B, 68] -> [B, 1404]
        offsets = torch.reshape(offsets, (-1, self.config['model']['vertex_num'], 3))
        reconstructed = subject_template + offsets
        return reconstructed
    
    def save(self, epoch: int, optimizer: torch.optim.Optimizer,
             scheduler: torch.optim.lr_scheduler.ExponentialLR,
             train_loss_hist: 'list[float]', val_loss_hist: 'list[float]'):
        save_dict = {
            'model': self.state_dict(),
            'optim': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'train_loss_hist': train_loss_hist,
            'val_loss_hist': val_loss_hist
        }
        file_name = f'AGK_{epoch}.pt'
        save_torch(save_dict, file_name=file_name, dir_path='training')

    def load(self, epoch: int) -> tuple[dict, dict, list[float], list[float]]:
        file_name = f'AGK_{epoch}.pt'
        file_path = os.path.join('training', file_name)
        try:
            state = load_torch(file_path)
            self.load_state_dict(state['model'])
            return state['optim'], state['scheduler'], state['train_loss_hist'], state['val_loss_hist']
        except ValueError:
            print(f'No file at processed_data/training with name {file_name}.')
            return None, None, None, None


class AudioAnalyzer(nn.Module):
    def __init__(self, input_type: str = 'melspec'):
        super(AudioAnalyzer, self).__init__()

        if input_type == 'melspec':
            self.net = nn.Sequential(
                nn.Conv2d(in_channels=9, out_channels=16, kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)),
                # [B, 9, 128, 30] -> [B, 16, 64, 30]
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16,  kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)),
                # [B, 16, 64, 30] -> [B, 16, 32, 30]
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32,  kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)),
                # [B, 16, 32, 30] -> [B, 32, 16, 30]
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32,  kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)),
                # [B, 32, 16, 30] -> [B, 32, 8, 30]
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32,  kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)),
                # [B, 32, 8, 30] -> [B, 32, 4, 30]
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64,  kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)),
                # [B, 32, 4, 30] -> [B, 64, 2, 30]
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64,  kernel_size=(2, 1), stride=(2, 1)),
                # [B, 64, 2, 30] -> [B, 64, 1, 30]
                nn.ReLU(),
            )
        else:   # for MFCC
            self.net = nn.Sequential(
                nn.Conv2d(in_channels=9, out_channels=16, kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)),
                # [B, 13, 40, 30] -> [B, 16, 20, 30]
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=16,  kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)),
                # [B, 16, 20, 30] -> [B, 16, 10, 30]
                nn.ReLU(),
                nn.Conv2d(in_channels=16, out_channels=32,  kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)),
                # [B, 16, 10, 30] -> [B, 32, 5, 30]
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1), padding=(1, 0), stride=(2, 1)),
                # [B, 32, 5, 30] -> [B, 32, 3, 30]
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2, 1), stride=(2, 1)),
                # [B, 32, 3, 30] -> [B, 64, 1, 30]
                nn.ReLU(),
            )
        
    def forward(self, audio_feature: torch.Tensor):
        return self.net(audio_feature)


class Articulation(nn.Module):
    def __init__(self):
        super(Articulation, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=68, out_channels=64,  kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)),
            # [B, 68, 1, 30] -> [B, 64, 1, 15]
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,  kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)),
            # [B, 64, 2, 15] -> [B, 64, 1, 8]
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,  kernel_size=(1, 3), padding=(0, 1), stride=(1, 2)),
            # [B, 64, 2, 8] -> [B, 64, 1, 4]
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64,  kernel_size=(1, 4), stride=(1, 4)),
            # [B, 64, 2, 4] -> [B, 64, 1, 1]
            nn.ReLU(),
        )

    def forward(self, audio_feature: torch.tensor):
        return self.net(audio_feature)
