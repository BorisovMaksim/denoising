import uuid
import ffmpeg
import gradio as gr
from pathlib import Path
from denoisers.SpectralGating import SpectralGating
from huggingface_hub import hf_hub_download
from denoisers.demucs import Demucs
import torch
import torchaudio
import yaml


def denoising_transform(audio, model):
    src_path = Path("cache_wav/original/{}.wav".format(str(uuid.uuid4())))
    tgt_path = Path("cache_wav/denoised/{}.wav".format(str(uuid.uuid4())))
    src_path.parent.mkdir(exist_ok=True, parents=True)
    tgt_path.parent.mkdir(exist_ok=True, parents=True)
    (ffmpeg.input(audio)
     .output(src_path.as_posix(), acodec='pcm_s16le', ac=1, ar=22050)
     .run()
     )
    wav, rate = torchaudio.load(audio)
    reduced_noise = model.predict(wav)
    torchaudio.save(tgt_path, reduced_noise, rate)
    return tgt_path


def run_app(model_filename, config_filename):
    model_path = hf_hub_download(repo_id="BorisovMaksim/demucs", filename=model_filename)
    config_path = hf_hub_download(repo_id="BorisovMaksim/demucs", filename=config_filename)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    model = Demucs(config['demucs'])
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])

    interface_demucs = gr.Interface(
        fn=lambda x: denoising_transform(x, model),
        inputs=gr.Audio(label="Source Audio", source="microphone", type='filepath'),
        outputs=gr.Audio(label="Demucs", type='filepath')
    )
    interface_spectral_gating = gr.Interface(
        fn=lambda x: denoising_transform(x, SpectralGating()),
        inputs=gr.Audio(label="Source Audio", source="microphone", type='filepath'),
        outputs=gr.Audio(label="Spectral Gating", type='filepath')
    )
    gr.Parallel(interface_demucs, interface_spectral_gating,
                title="Denoising",
                examples=[
                    ["testing/wavs/p232_071.wav"],
                    ["testing/wavs/p232_284.wav"],
                ]).launch()


if __name__ == "__main__":
    model_filename = "original_sr/Demucs_original_sr_epoch3.pt"
    config_filename = "original_sr/config.yaml"
    run_app(model_filename, config_filename)
