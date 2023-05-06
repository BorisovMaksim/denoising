import os
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from omegaconf import DictConfig
import wandb
import torchaudio

from checkpoing_saver import CheckpointSaver
from denoisers import get_model
from optimizers import get_optimizer
from losses import get_loss
from datasets import get_datasets
from testing.metrics import Metrics
import omegaconf

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(cfg: DictConfig):
    wandb.login(key=cfg['wandb']['api_key'], host=cfg['wandb']['host'])
    wandb.init(project=cfg['wandb']['project'],
               notes=cfg['wandb']['notes'],
               tags=cfg['wandb']['tags'],
               config=omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True))

    checkpoint_saver = CheckpointSaver(dirpath=cfg['training']['model_save_path'])
    metrics = Metrics(rate=cfg['dataloader']['sample_rate'])

    model = get_model(cfg['model']).to(device)
    optimizer = get_optimizer(model.parameters(), cfg['optimizer'])
    loss_fn = get_loss(cfg['loss'])
    train_dataset, valid_dataset = get_datasets(cfg)

    training_loader = DataLoader(train_dataset, batch_size=cfg['dataloader']['train_batch_size'], shuffle=True)
    validation_loader = DataLoader(valid_dataset, batch_size=cfg['dataloader']['valid_batch_size'], shuffle=True)

    wandb.watch(model, log_freq=100)

    for epoch in range(cfg['training']['num_epochs']):
        model.train(True)
        for i, data in enumerate(training_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            if i % cfg['wandb']['log_interval'] == 0:
                wandb.log({"loss": loss})

        model.train(False)

        running_vloss, running_pesq, running_stoi = 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
                running_metrics = metrics.calculate(denoised=voutputs, clean=vlabels)
                running_pesq += running_metrics['PESQ']
                running_stoi += running_metrics['STOI']


            avg_vloss = running_vloss / len(validation_loader)
            avg_pesq = running_pesq / len(validation_loader)
            avg_stoi = running_stoi / len(validation_loader)

            wandb.log({"valid_loss": avg_vloss,
                       "valid_pesq": avg_pesq,
                       "valid_stoi": avg_stoi})

            for tag, wav_path in cfg['validation']['wavs'].items():
                wav, rate = torchaudio.load(Path(cfg['validation']['path']) / wav_path)
                wav = torch.reshape(wav, (1, 1, -1)).to(device)
                prediction = model(wav)
                wandb.log({
                    f"{tag}_epoch_{epoch}": wandb.Audio(
                        prediction.cpu()[0][0],
                        sample_rate=rate)})

            checkpoint_saver(model, epoch, metric_val=avg_pesq)


if __name__ == '__main__':
    train()
