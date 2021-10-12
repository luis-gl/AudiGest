import torch

from torch import nn


class EmotionEncoder(nn.Module):
    def __init__(self):
        super(EmotionEncoder, self).__init__()

        self.encoder = nn.Sequential(
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

    def forward(self, melspec: torch.Tensor):
        return self.encoder(melspec)  # -> [B, 16, 3, 2]