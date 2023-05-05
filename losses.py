import torch

LOSSES = {
    'mse': torch.nn.MSELoss()
}


def get_loss(loss_config):
    return LOSSES[loss_config['name']]
