import torch

from torch import nn


class LanguageExtractor(nn.Module):
    def __init__(self, config: dict):
        super(LanguageExtractor, self).__init__()

        self.config = config

        self.input_dim = config['audio']['n_mfcc']
        self.hidden_dim = config['model']['lstm']['hidden_dim']
        self.layers = config['model']['lstm']['num_layers']

        self.lang_encoder = nn.LSTM(input_size=self.input_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=self.layers,
                                    batch_first=True)
                            # -> [B, L, 16]: Batch, Length, hidden_size
        
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=config['audio']['fps']*self.hidden_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128)
        )
    
    def forward(self, mfcc: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]):
        lang, (h, c) = self.lang_encoder(mfcc, hidden)      # -> [B, 30, 16], ([1, B, 16], [1, B, 16])
        lang = self.features(lang)                          # -> [B, 128]
        h, c = h.detach(), c.detach()
        return lang, (h, c)