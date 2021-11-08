from torch import nn


class VelocityLoss(nn.Module):
    def __init__(self, config: dict, rec_loss):
        super(VelocityLoss, self).__init__()
        self.weight = config['model']['velocity_weight']
        self.vertex_num = config['model']['vertex_num']
        self.reconstruction_loss = rec_loss

    def forward(self, predicted, target):
        # predicted, target -> [B, L, 468, 3]
        if self.weight > 0:
            pred1 = predicted[:, :, :-1, :]
            pred2 = predicted[:, :, 1:, :]
            velocity_pred = pred2 - pred1

            target1 = target[:, :, :-1, :]
            target2 = target[:, :, 1:, :]
            velocity_real = target2 - target1

            velocity_loss = self.weight * self.reconstruction_loss(velocity_pred, velocity_real)
            return velocity_loss
        else:
            return 0.0
