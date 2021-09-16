import os
import torch

from torch import nn
from utils.files.save import load_torch, save_torch


class AudiGest(nn.Module):
    def __init__(self, config: dict):
        super(AudiGest, self).__init__()

        self.config = config

        self.emotion_encoder = nn.Sequential(
            nn.BatchNorm2d(1, eps=1e-5),
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5)),
            # [B, 1, 128, 31] -> [B, 8, 124, 27]
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5, 5)),
            # [B, 8, 124, 27] -> [B, 8, 120, 23]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(3, 2)),
            # [B, 8, 120, 23] -> [B, 8, 40, 11]
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(3, 2)),
            # [B, 8, 40, 11] -> [B, 16, 13, 5]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=(5, 2)),
            # [B, 16, 13, 5] -> [B, 16, 3, 2]
        )

        self.input_dim = config['audio']['n_mfcc']
        self.hidden_dim = config['model']['lstm']['hidden_dim']
        self.layers = config['model']['lstm']['num_layers']
        self.lang_encoder = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim,
                                    num_layers=self.layers, batch_first=True)
        # -> [B, L, 32]: Batch, Length, hidden_size

        fc_config = config['model']['fc']
        self.decoder = nn.Sequential(
            nn.Linear(in_features=fc_config['in_feat'], out_features=fc_config['hidden_1']),
            nn.ReLU(),
            nn.Dropout(fc_config['drop']),
            nn.Linear(in_features=fc_config['hidden_1'], out_features=fc_config['hidden_2']),
            nn.ReLU(),
            nn.Dropout(fc_config['drop']),
            nn.Linear(in_features=fc_config['hidden_2'], out_features=fc_config['hidden_3']),
            nn.ReLU(),
            nn.Dropout(fc_config['drop']),
            nn.Linear(in_features=fc_config['hidden_3'], out_features=fc_config['hidden_4']),
            nn.Tanh(),
            nn.Linear(in_features=fc_config['hidden_4'], out_features=config['model']['vertex_num']*3)
        )

    def forward(self, melspec: torch.Tensor, mfcc: torch.Tensor, hidden: torch.Tensor=None):
        encoded_em = self.emotion_encoder(melspec)  # -> [B, 16, 3, 2]
        encoded_em = encoded_em.flatten(start_dim=1)
        lstm_res, hidden = self.lang_encoder(mfcc, hidden)
        h, c = hidden
        h, c = h.detach(), c.detach()
        hidden = (h, c)
        # -> [B, 30, 16], [1, B, 16], [1, B, 16]
        lstm_res = lstm_res.flatten(start_dim=1)
        concat = torch.cat((lstm_res, encoded_em), 1)
        reconstructed = self.decoder(concat)
        reconstructed = torch.reshape(reconstructed, (-1, self.config['model']['vertex_num'], 3))
        return reconstructed, hidden

    def save(self, epoch: int, optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.ExponentialLR,
            train_loss_hist: list[float], val_loss_hist: list[float]):

        save_dict = {
            'model': self.state_dict(),
            'optim': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss_hist': train_loss_hist,
            'val_loss_hist': val_loss_hist
        }
        file_name = f'AG_{epoch}.pt'
        save_torch(save_dict, file_name=file_name, dir_path='training')

    def load(self, epoch: int) -> tuple[dict, dict, list[float], list[float]]:
        file_name = f'AG_{epoch}.pt'
        file_path = os.path.join('training', file_name)
        state = None
        try:
            state = load_torch(file_path)
            self.load_state_dict(state['model'])
            return state['optim'], state['scheduler'], state['train_loss_hist'], state['val_loss_hist']
        except ValueError:
            print(f'No file at processed_data/training with name {file_name}, starting training from 0.')
            return None, None, None, None
