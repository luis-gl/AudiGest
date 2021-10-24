import torch

from torch import nn


class SpeechExtractor(nn.Module):
    def __init__(self, config: dict):
        super(SpeechExtractor, self).__init__()

        self.config = config

        self.input_dim = config['audio']['n_mfcc'] * 2
        self.hidden_dim = config['model']['lstm']['hidden_dim']
        self.layers = config['model']['lstm']['num_layers']

        self.lang_encoder = nn.LSTM(input_size=self.input_dim,
                                    hidden_size=self.hidden_dim,
                                    num_layers=self.layers,
                                    batch_first=True)
        # -> [B, L, 16]: Batch, Length, hidden_size
    
    def forward(self, mfcc: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]):
        lang, (h, c) = self.lang_encoder(mfcc, hidden)
        # -> [B, 30, 16], ([1, B, 16], [1, B, 16])
        h, c = h.detach(), c.detach()
        return lang, (h, c)