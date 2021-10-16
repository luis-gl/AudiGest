import torch
from torch import nn


class VelocityLoss(nn.Module):
    def __init__(self, config: dict, rec_loss):
        super(VelocityLoss, self).__init__()
        self.weight = config['model']['velocity_weight']
        self.consecutive_seqs = config['training']['consecutive_seqs']
        self.vertex_num = config['model']['vertex_num']
        self.reconstruction_loss = rec_loss

    def forward(self, predicted, target):
        if self.weight > 0 and self.consecutive_seqs >= 2:
            verts_predicted = torch.reshape(predicted, (-1, self.consecutive_seqs, self.vertex_num, 3))
            x1_pred = torch.reshape(verts_predicted[:, -1, :], (-1, self.vertex_num, 3))
            x2_pred = torch.reshape(verts_predicted[:, -2, :], (-1, self.vertex_num, 3))
            velocity_pred = x1_pred - x2_pred

            verts_target = torch.reshape(target, (-1, self.consecutive_seqs, self.vertex_num, 3))
            x1_target = torch.reshape(verts_target[:, -1, :], (-1, self.vertex_num, 3))
            x2_target = torch.reshape(verts_target[:, -2, :], (-1, self.vertex_num, 3))
            velocity_real = x1_target - x2_target

            velocity_loss = self.weight * self.reconstruction_loss(velocity_pred, velocity_real)
            return velocity_loss
        else:
            return 0.0
