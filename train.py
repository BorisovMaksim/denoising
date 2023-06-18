import os
import torch
from torch.utils.data import DataLoader
import omegaconf
from omegaconf import DictConfig
import wandb
from torch.optim.lr_scheduler import ExponentialLR

from checkpoing_saver import CheckpointSaver
from denoisers import get_model
from optimizers import get_optimizer
from losses import get_loss
from datasets import get_datasets
from testing.metrics import Metrics
from datasets.minimal import Minimal
from tqdm import tqdm
from pathlib import Path


def train(cfg: DictConfig):
    device = torch.device(f'cuda:{cfg.gpu}' if torch.cuda.is_available() else 'cpu')
    wandb.login(key=cfg['wandb']['api_key'], host=cfg['wandb']['host'])
    wandb.init(project=cfg['wandb']['project'],
               notes=cfg['wandb']['notes'],
               config=omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
               resume=cfg['wandb']['resume'],
               name=cfg['wandb']['run_name'])
    checkpoint_saver = CheckpointSaver(dirpath=cfg['training']['model_save_path'], run_name=wandb.run.name,
                                       decreasing=False)
    metrics = Metrics(source_rate=cfg['dataloader']['sample_rate']).to(device)

    model = get_model(cfg['model']).to(device)
    optimizer = get_optimizer(model.parameters(), cfg['optimizer'])
    scheduler = ExponentialLR(optimizer, gamma=0.9)
    loss_fn = get_loss(cfg['loss'], device)

    train_dataset, valid_dataset = get_datasets(cfg)
    minimal_dataset = Minimal(cfg)

    dataloaders = {
        'train':  DataLoader(train_dataset, batch_size=cfg['dataloader']['train_batch_size'], shuffle=True,
                             num_workers=cfg['dataloader']['num_workers']),
        'val': DataLoader(valid_dataset, batch_size=cfg['dataloader']['valid_batch_size'], shuffle=False,
                          num_workers=cfg['dataloader']['num_workers']),
        'minimal': DataLoader(minimal_dataset)
    }

    wandb.watch(model, log_freq=cfg['wandb']['log_interval'])
    epoch = 0

    if cfg['wandb']['resume_path'] is not None:
        checkpoint = torch.load(cfg['wandb']['resume_path'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        print(f"Continue from {cfg['wandb']['resume_path']}, epoch={epoch}, loss={loss}")


    while epoch < cfg['training']['num_epochs']:
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss, running_pesq, running_stoi = 0.0, 0.0, 0.0
            loop = tqdm(dataloaders[phase])
            for i, (inputs, labels) in enumerate(loop):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_metrics = metrics(denoised=outputs, clean=labels)
                running_loss += loss.item() * inputs.size(0)
                running_pesq += running_metrics['PESQ']
                running_stoi += running_metrics['STOI']

                loop.set_description(f"Epoch [{epoch}/{cfg['training']['num_epochs']}][{phase}]")
                loop.set_postfix(loss=running_loss / (i + 1) / inputs.size(0),
                                 pesq=running_pesq / (i + 1) / inputs.size(0),
                                 stoi=running_stoi / (i + 1) / inputs.size(0))

                if phase == 'train' and i % cfg['wandb']['log_interval'] == cfg['wandb']['log_interval'] - 1:
                    wandb.log({"train_loss": running_loss / (i + 1) / inputs.size(0),
                               "train_pesq": running_pesq / (i + 1) / inputs.size(0),
                               "train_stoi": running_stoi / (i + 1) / inputs.size(0)})

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            eposh_pesq = running_pesq / len(dataloaders[phase].dataset)
            eposh_stoi = running_stoi / len(dataloaders[phase].dataset)

            wandb.log({f"{phase}_loss": epoch_loss,
                       f"{phase}_pesq": eposh_pesq,
                       f"{phase}_stoi": eposh_stoi})

            if phase == 'val':
                for i, (wav, rate) in enumerate(dataloaders['minimal']):
                    if cfg['dataloader']['normalize']:
                        scale = torch.std(wav)
                        wav = wav / scale
                        prediction = model(wav.to(device))
                        prediction = prediction * scale
                    else:
                        prediction = model(wav.to(device))
                    wandb.log({
                        f"{i}_example": wandb.Audio(
                            prediction.detach().cpu().numpy()[0][0],
                            sample_rate=rate)})

                checkpoint_saver(model, epoch, metric_val=eposh_pesq,
                                 optimizer=optimizer, loss=epoch_loss)
            else:
                scheduler.step()
        epoch += 1


if __name__ == "__main__":
    pass
