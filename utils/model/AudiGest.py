import torch
from torch import nn


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

        self.input_s = config['audio']['n_mfcc']
        self.hidden_s = config['model']['lstm']['hidden_dim']
        self.layers = config['model']['lstm']['num_layers']
        self.lang_encoder = nn.LSTM(input_size=self.input_s, hidden_size=self.hidden_s,
                                    num_layers=self.layers, batch_first=True)
        # -> [B, L, 32]: Batch, Length, hidden_size
        self.hidden = None
        self.cell = None

        self.decoder = nn.Sequential(
            nn.Linear(in_features=config['model']['fc']['in_feat'], out_features=config['model']['fc']['hidden_1']),
            nn.ReLU(),
            nn.Dropout(config['model']['fc']['drop']),
            nn.Linear(in_features=config['model']['fc']['hidden_1'], out_features=config['model']['fc']['hidden_2']),
            nn.Tanh(),
            nn.Dropout(config['model']['fc']['drop']),
            nn.Linear(in_features=config['model']['fc']['hidden_2'], out_features=config['model']['vertex_num']*3)
        )

    def init_state(self):
        self.hidden = torch.empty(self.config['model']['lstm']['num_layers'], self.config['training']['batch_size'],
                                  self.config['model']['lstm']['hidden_dim'], dtype=torch.float32)
        self.cell = torch.empty(self.config['model']['lstm']['num_layers'], self.config['training']['batch_size'],
                                  self.config['model']['lstm']['hidden_dim'], dtype=torch.float32)
        self.hidden = nn.init.constant_(self.hidden, 0.0)
        self.cell = nn.init.constant_(self.cell, 0.0)

    def forward(self, melspec, mfcc):
        encoded_em = self.emotion_encoder(melspec)  # -> [B, 16, 3, 2]
        encoded_em = encoded_em.flatten(start_dim=1)
        lstm_res, (self.hidden, self.cell) = self.lang_encoder(mfcc, (self.hidden, self.cell))
        # -> [B, 30, 16], [1, B, 16], [1, B, 16]
        self.hidden, self.cell = self.hidden.permute(1, 0, 2), self.cell.permute(1, 0, 2)
        self.hidden = self.hidden.flatten(start_dim=1)
        self.cell = self.cell.flatten(start_dim=1)
        concat = torch.cat((self.hidden, self.cell, encoded_em), 1)
        reconstructed = self.decoder(concat)
        reconstructed = torch.reshape(reconstructed, (-1, self.config['model']['vertex_num'], 3))
        return reconstructed
