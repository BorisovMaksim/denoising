import argparse
import os

from tqdm import tqdm

from utils.utils import collect_valentini_paths
from metrics import Metrics
from denoisers.SpectralGating import SpectralGating
import torch
import torchaudio
import yaml
from denoisers.demucs import Demucs

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

PARSERS = {
    'valentini': collect_valentini_paths
}
MODELS = {
    'baseline': SpectralGating,
    'demucs': Demucs
}



def evaluate_on_dataset(model_name, dataset_path, dataset_type, model_path):
    if model_name is not None:
        with open("conf/model/demucs.yaml", "r") as f:
            model_cfg = yaml.safe_load(f)
        model_cfg['demucs']['L'] = 5

        model = MODELS[model_name](model_cfg['demucs'])
        checkpoint = torch.load(model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint['model_state_dict'])


    parser = PARSERS[dataset_type]
    clean_wavs, noisy_wavs = parser(dataset_path)

    metrics = Metrics(source_rate=48000)
    mean_scores = {'PESQ': 0, 'STOI': 0}
    for clean_path, noisy_path in tqdm(zip(clean_wavs, noisy_wavs), total=len(clean_wavs)):
        clean_wav, rate = torchaudio.load(clean_path)
        noisy_wav, rate = torchaudio.load(noisy_path)

        if model_name is None:
            scores = metrics(denoised=noisy_wav, clean=clean_wav)
        else:
            denoised_wav = model.predict(noisy_wav)
            scores = metrics(denoised=denoised_wav, clean=clean_wav)

        mean_scores['PESQ'] += scores['PESQ']
        mean_scores['STOI'] += scores['STOI']

    mean_scores['PESQ'] = mean_scores['PESQ'] / len(clean_wavs)
    mean_scores['STOI'] = mean_scores['STOI'] / len(clean_wavs)

    return mean_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Program to evaluate denoising')
    parser.add_argument('--dataset_path', type=str,
                        default='/media/public/data/denoising/DS_10283_2791/',
                        help='Path to dataset folder')
    parser.add_argument('--dataset_type', type=str, required=True,
                        choices=['valentini'])
    parser.add_argument('--model_name', type=str,
                        choices=['baseline', 'demucs'])
    parser.add_argument('--model_path', type=str)


    args = parser.parse_args()

    mean_scores = evaluate_on_dataset(model_name=args.model_name,
                        dataset_path=args.dataset_path,
                        dataset_type=args.dataset_type,
                        model_path=args.model_path)
    print(f"Metrics on {args.dataset_type} dataset with "
          f"{args.model_name if args.model_name is not None else 'ideal denoising'} = {mean_scores}")
