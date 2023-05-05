from denoisers.demucs import Demucs
from denoisers.SpectralGating import SpectralGating


MODEL_POOL = {
    'demucs': Demucs,
    'baseline': SpectralGating
}


def get_model(model_config):
    name, params = list(model_config.items())[0]
    return MODEL_POOL[name](params)

