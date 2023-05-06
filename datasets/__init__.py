from typing import Tuple
from torch.utils.data import Dataset

from datasets.valentini import Valentini
from transforms import Transform

DATASETS_POOL = {
    'valentini': Valentini
}



def get_datasets(cfg) -> Tuple[Dataset, Dataset]:
    name, dataset_params = list(cfg['dataset'].items())[0]
    transform = Transform(input_sr=dataset_params['sample_rate'], **cfg['dataloader'])
    train_dataset = DATASETS_POOL[name](valid=False, transform=transform, **dataset_params)
    valid_dataset = DATASETS_POOL[name](valid=True, transform=transform, **dataset_params)
    return train_dataset, valid_dataset
