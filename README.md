---
title: {{Denoising}}
emoji: {{🤗}}
colorFrom: {{red}}
colorTo: {{orange}}
sdk: {{gradio}}
sdk_version: {{ 3.28.1}}
app_file: app.py
pinned: false
---

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

