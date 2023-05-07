import os
import torch
from torch.utils.data import DataLoader
import omegaconf
from omegaconf import DictConfig
import wandb

from checkpoing_saver import CheckpointSaver
from denoisers import get_model
from optimizers import get_optimizer
from losses import get_loss
from datasets import get_datasets
from testing.metrics import Metrics
from datasets.minimal import Minimal

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(cfg: DictConfig):
    wandb.login(key=cfg['wandb']['api_key'], host=cfg['wandb']['host'])
    wandb.init(project=cfg['wandb']['project'],
               notes=cfg['wandb']['notes'],
               tags=cfg['wandb']['tags'],
               config=omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True))

    checkpoint_saver = CheckpointSaver(dirpath=cfg['training']['model_save_path'], run_name=wandb.run.name)
    metrics = Metrics(rate=cfg['dataloader']['sample_rate'])

    model = get_model(cfg['model']).to(device)
    optimizer = get_optimizer(model.parameters(), cfg['optimizer'])
    loss_fn = get_loss(cfg['loss'])
    train_dataset, valid_dataset = get_datasets(cfg)
    minimal_dataset = Minimal(cfg)

    dataloaders = {
        'train':  DataLoader(train_dataset, batch_size=cfg['dataloader']['train_batch_size'], shuffle=True),
        'val': DataLoader(valid_dataset, batch_size=cfg['dataloader']['valid_batch_size'], shuffle=True),
        'minimal': DataLoader(minimal_dataset)
    }

    wandb.watch(model, log_freq=100)

    for epoch in range(cfg['training']['num_epochs']):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_pesq, running_stoi = 0.0, 0.0, 0.0
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_metrics = metrics.calculate(denoised=outputs, clean=labels)
                running_loss += loss.item() * inputs.size(0)
                running_pesq += running_metrics['PESQ']
                running_stoi += running_metrics['STOI']

                if phase == 'train' and i % cfg['wandb']['log_interval'] == 0:
                    wandb.log({"train_loss": running_loss / (i + 1),
                               "train_pesq": running_pesq / (i + 1),
                               "train_stoi": running_stoi / (i + 1)})
            epoch_loss = running_loss / len(dataloaders[phase])
            eposh_pesq = running_pesq / len(dataloaders[phase])
            eposh_stoi = running_stoi / len(dataloaders[phase])

            wandb.log({f"{phase}_loss": epoch_loss,
                       f"{phase}_pesq": eposh_pesq,
                       f"{phase}_stoi": eposh_stoi})

            if phase == 'val':
                for i, (wav, rate) in enumerate(dataloaders['minimal']):
                    prediction = model(wav)
                    wandb.log({
                        f"{i}_example": wandb.Audio(
                            prediction.cpu()[0][0],
                            sample_rate=rate)})

                checkpoint_saver(model, epoch, metric_val=eposh_pesq)
