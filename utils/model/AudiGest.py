import os
from utils.model.LanguageExtractor import LanguageExtractor
import torch

from torch import nn
from utils.model.EmotionEncoder import EmotionEncoder
from utils.model.LanguageExtractor import LanguageExtractor
from utils.files.save import load_torch, save_torch


class AudiGest(nn.Module):
    def __init__(self, config: dict):
        super(AudiGest, self).__init__()

        self.config = config

        self.emotion_encoder = EmotionEncoder()
        self.lang_encoder = LanguageExtractor(config)

        fc_config = config['model']['fc']
        self.decoder = nn.Sequential(
            nn.Linear(in_features=136, out_features=fc_config['hidden_1']),
            nn.ReLU(),
            nn.Dropout(fc_config['drop']),
            nn.Linear(in_features=fc_config['hidden_1'], out_features=fc_config['hidden_2']),
            nn.Tanh(),
            nn.Linear(in_features=fc_config['hidden_2'], out_features=config['model']['vertex_num']*3)
        )

    def forward(self, melspec: torch.Tensor, mfcc: torch.Tensor, base_target: torch.Tensor,
                hidden: tuple[torch.Tensor, torch.Tensor]=None):
        em = self.emotion_encoder(melspec)              # -> [B, 8]
        lang, hidden = self.lang_encoder(mfcc, hidden)  # -> [B, 128]
        concat = torch.cat((lang, em), 1)               # -> [B, 136]
        offset = self.decoder(concat)            # -> [B, 1404]
        offset = torch.reshape(offset, (-1, self.config['model']['vertex_num'], 3))
        reconstructed = base_target + offset
        return reconstructed, hidden    # -> [B, 468, 3], ([1, B, 16], [1, B, 16])

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
            print(f'No file at processed_data/training with name {file_name}.')
            return None, None, None, None
