import os
import torch

from torch import nn
from utils.files.save import load_torch, save_torch


class SequenceRegressor(nn.Module):
    def __init__(self, config: dict, device: torch.device, feature_type: str = 'melspec'):
        super(SequenceRegressor, self).__init__()

        self.config = config
        self.device = device

        if config['model']['use_condition']:
            self.condition_num = 12
        else:
            self.condition_num = 0

        if feature_type == 'melspec':
            self.input_dim = 128
        else:
            self.input_dim = config['audio']['n_mfcc'] * 2
        self.hidden_dim = config['model']['hidden_dim']
        self.layers = config['model']['num_layers']

        self.seq_modeler = nn.LSTM(input_size=self.input_dim + self.condition_num,
                                   hidden_size=self.hidden_dim,
                                   num_layers=self.layers,
                                   batch_first=True)
        # [B, L, melspec/mfcc] -> [B, L, hidden_dim]

        self.offsets_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_dim + self.condition_num, out_features=config['model']['vertex_num'] * 3),
            nn.Tanh()
        )

    def forward(self, feature: torch.Tensor, emotion: torch.Tensor, subject: torch.Tensor, subject_template):
        h_0 = torch.zeros(self.layers, feature.shape[0], self.hidden_dim).to(self.device)
        c_0 = torch.zeros(self.layers, feature.shape[0], self.hidden_dim).to(self.device)

        if self.condition_num > 0:
            subject = torch.tile(subject, (1, feature.shape[1], 1))
            emotion = torch.tile(emotion, (1, feature.shape[1], 1))
            feature = torch.cat((subject, emotion, feature), dim=-1)

        seq_features, _ = self.seq_modeler(feature, (h_0, c_0))
        # -> [B, L, hidden_dim], ([layers, B, hidden], [layers, B, hidden])

        if self.condition_num > 0:
            seq_features = torch.cat((subject, emotion, seq_features), dim=-1)

        offsets = self.offsets_layer(seq_features)
        offsets = offsets.reshape(-1, offsets.shape[1], self.config['model']['vertex_num'], 3)
        reconstructed = subject_template + offsets
        return reconstructed

    def save(self, epoch, optimizer, scheduler, train_loss_hist, val_loss_hist):

        save_dict = {
            'model': self.state_dict(),
            'optim': optimizer.state_dict(),
            'scheduler': scheduler.state_dict() if scheduler is not None else None,
            'train_loss_hist': train_loss_hist,
            'val_loss_hist': val_loss_hist
        }
        file_name = f'AG_{epoch}.pt'
        save_torch(save_dict, file_name=file_name, dir_path='training')

    def load(self, epoch: int):
        file_name = f'AG_{epoch}.pt'
        file_path = os.path.join('training', file_name)
        try:
            state = load_torch(file_path)
            self.load_state_dict(state['model'])
            return state['optim'], state['scheduler'], state['train_loss_hist'], state['val_loss_hist']
        except ValueError:
            print(f'No file at processed_data/training with name {file_name}.')
            return None, None, None, None
