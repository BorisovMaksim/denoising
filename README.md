---
title: Denoising
emoji: 🤗
colorFrom: red
colorTo: orange
sdk: gradio
sdk_version: 3.28.1
app_file: app.py
pinned: false
---

# Experiments

| Experiment | Description | Result                                                 |
|--------------|:-----:|--------------------------------------------------------|
| Baseline | Initial experiment with L1 loss  | Poor quality                                           |
| Baseline_L1_Multi_STFT_loss     |  Changed loss to Multi STFT + L1 loss | Better performance                                     | 
|L1_Multi_STFT_no_resample  | Tried to train without resampling | No impovement, probably because RELU on the last layer |
|Updated_DEMUCS | Used relu in the last layer. Removed it.| Significant improvement                                |
|wav_normalization | Tried to normalized wav by std during training| Small improvement                                      |
| original_sr| Train with original sample rate | Significant improvement                                |
|increased_L | Increased number of encoder-decoder pairs from 3 to 5| Performance comparable with original_sr                |
| double_sr| Train with double sample rate| in progress                                            | 

![img.png](images/img.png)
# MVP
Сервисом является web interface, в котором пользователь 
сможет записать своей голос в шумных условиях и получить на выход аудиозапись без шума.
Для обработки шумных аудио файлов есть доступ к  API на питоне.

Web interface реализован на gradio. Сама работа пишется в контексте фрейморка pytorch.
В качестве системы контроля экспериментов выбран wandb. Для управления конфигами - hydra.
Архитектура модели базируется на работе "Real Time Speech Enhancement in the Waveform Domain" от facebook.



# Testing
|                 | valentini_PESQ | valentini_STOI |
|:---------------:|:--------------:|:--------------:|
| ideal denoising |     1.9709     |     0.9211     |
|    baseline     |     1.7433     |     0.8844     |

