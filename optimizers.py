import torch

OPTIMIZERS_POOL = {
    'sgd': torch.optim.SGD,
    'adam': torch.optim.Adam
}

def get_optimizer(model_params, optimizer_config):
    name, params = list(optimizer_config.items())[0]
    optimizer = OPTIMIZERS_POOL[name](model_params, **params)
    return optimizer
