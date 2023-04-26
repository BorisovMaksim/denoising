import torchaudio
import torch
import matplotlib.pyplot as plt
from pathlib import Path


def collect_valentini_paths(dataset_path):
    clean_path = Path(dataset_path) / 'clean_testset_wav'
    noisy_path = Path(dataset_path) / 'noisy_testset_wav'

    clean_wavs = list(clean_path.glob("*"))
    noisy_wavs = list(noisy_path.glob("*"))

    return clean_wavs, noisy_wavs


def load_wav(path):
    wav, org_sr = torchaudio.load(path)
    wav = torchaudio.functional.resample(wav, orig_freq=org_sr, new_freq=16000)
    return wav



def plot_spectrogram(stft, title="Spectrogram", xlim=None):
    magnitude = stft.abs()
    spectrogram = 20 * torch.log10(magnitude + 1e-8).numpy()
    figure, axis = plt.subplots(1, 1)
    img = axis.imshow(spectrogram, cmap="viridis", vmin=-100, vmax=0, origin="lower", aspect="auto")
    figure.suptitle(title)
    plt.colorbar(img, ax=axis)
    plt.show()


def plot_mask(mask, title="Mask", xlim=None):
    mask = mask.numpy()
    figure, axis = plt.subplots(1, 1)
    img = axis.imshow(mask, cmap="viridis", origin="lower", aspect="auto")
    figure.suptitle(title)
    plt.colorbar(img, ax=axis)
    plt.show()


    

def generate_mixture(waveform_clean, waveform_noise, target_snr):
    
    power_clean_signal = waveform_clean.pow(2).mean()
    power_noise_signal = waveform_noise.pow(2).mean()
    current_snr = 10 * torch.log10(power_clean_signal / power_noise_signal)
    
    waveform_noise *= 10 ** (-(target_snr - current_snr) / 20)
    return waveform_clean + waveform_noise

