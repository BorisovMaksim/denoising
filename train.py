import os
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import Sequential
from torch.utils.data import DataLoader
from datetime import datetime
from torchvision.transforms import RandomCrop
from utils import load_wav
from denoisers.demucs import Demucs
from pathlib import Path
from omegaconf import DictConfig

from optimizers import OPTIMIZERS_POOL
from losses import LOSSES
from datasets import DATASETS_POOL
from denoisers import get_model
from optimizers import get_optimizer
from losses import get_loss


os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#
#
# DATASET_PATH = Path('/media/public/dataset/denoising/DS_10283_2791/')
# VALID_WAVS = {'hard': 'p257_171.wav',
#               'medium': 'p232_071.wav',
#               'easy': 'p232_284.wav'}
# MAX_SECONDS = 2
# SAMPLE_RATE = 16000
#
# transform = Sequential(RandomCrop((1, int(MAX_SECONDS * SAMPLE_RATE)), pad_if_needed=True))
#
# training_loader = DataLoader(Valentini(valid=False, transform=transform), batch_size=12, shuffle=True)
# validation_loader = DataLoader(Valentini(valid=True, transform=transform), batch_size=12, shuffle=True)



def train(cfg: DictConfig):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model = get_model(cfg['model'])
    optimizer = get_optimizer(model.parameters(), cfg['optimizer'])
    loss_fn = get_loss(cfg['loss'])

    writer = SummaryWriter('runs/denoising_trainer_{}'.format(timestamp))
    epoch_number = 0

    EPOCHS = 5

    best_vloss = 1_000_000.

    for tag, wav_path in VALID_WAVS.items():
        wav = load_wav(DATASET_PATH / 'noisy_testset_wav' / wav_path)
        writer.add_audio(tag=tag, snd_tensor=wav, sample_rate=SAMPLE_RATE)
    writer.flush()

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)

        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(training_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000  # loss per batch
                print('  batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_number * len(training_loader) + i + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        avg_loss = last_loss

        model.train(False)

        running_vloss = 0.0
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            writer.add_scalars('Training vs. Validation Loss',
                               {'Training': avg_loss, 'Validation': avg_vloss},
                               epoch_number + 1)
            for tag, wav_path in VALID_WAVS.items():
                wav = load_wav(DATASET_PATH / 'noisy_testset_wav' / wav_path)
                wav = torch.reshape(wav, (1, 1, -1)).to(device)
                prediction = model(wav)
                writer.add_audio(tag=f"Model predicted {tag} on epoch {epoch}",
                                 snd_tensor=prediction,
                                 sample_rate=SAMPLE_RATE)
            writer.flush()

            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'checkpoints/model_{}_{}'.format(timestamp, epoch_number)
                torch.save(model.state_dict(), model_path)

            epoch_number += 1


if __name__ == '__main__':
    train()
