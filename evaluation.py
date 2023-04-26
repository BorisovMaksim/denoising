import argparse
from tqdm import tqdm

from utils import load_wav, collect_valentini_paths
from metrics import Metrics
from denoisers.SpectralGating import SpectralGating


PARSERS = {
    'valentini': collect_valentini_paths
}
MODELS = {
    'baseline': SpectralGating
}



def evaluate_on_dataset(model_name, dataset_path, dataset_type, ideal):
    model = MODELS[model_name]()
    parser = PARSERS[dataset_type]
    clean_wavs, noisy_wavs = parser(dataset_path)

    metrics = Metrics()
    mean_scores = {'PESQ': 0, 'STOI': 0}
    for clean_path, noisy_path in tqdm(zip(clean_wavs, noisy_wavs), total=len(clean_wavs)):
        clean_wav = load_wav(clean_path)
        noisy_wav = load_wav(noisy_path)
        denoised_wav = model(noisy_wav)
        if ideal:
            scores = metrics.calculate(noisy_wav, clean_wav)
        else:
            scores = metrics.calculate(noisy_wav, denoised_wav)

        mean_scores['PESQ'] += scores['PESQ']
        mean_scores['STOI'] += scores['STOI']

    mean_scores['PESQ'] = mean_scores['PESQ'].numpy() / len(clean_wavs)
    mean_scores['STOI'] = mean_scores['STOI'].numpy() / len(clean_wavs)

    return mean_scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Program to evaluate denoising')
    parser.add_argument('--dataset_path', type=str,
                        default='/media/public/datasets/denoising/DS_10283_2791/',
                        help='Path to dataset folder')
    parser.add_argument('--dataset_type', type=str, required=True,
                        choices=['valentini'])
    parser.add_argument('--model_name', type=str,
                        choices=['baseline'])
    parser.add_argument('--ideal', type=bool, default=False,
                        help="Evaluate metrics on testing data with ideal denoising")

    args = parser.parse_args()

    mean_scores = evaluate_on_dataset(model_name=args.model_name,
                        dataset_path=args.dataset_path,
                        dataset_type=args.dataset_type,
                        ideal=args.ideal)
    print(f"Metrics on {args.dataset_type} dataset with "
          f"{args.model_name if args.model_name is not None else 'ideal denoising'} = {mean_scores}")
