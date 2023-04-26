import sys
import os

from re import M
import uuid
import shutil
import ffmpeg
import logging
import gradio as gr


from denoisers.SpectralGating import SpectralGating

model = SpectralGating()


def denoising_transform(audio):
    src_path = "cache_wav/source/{}.wav".format(str(uuid.uuid4()))
    tgt_path = "cache_wav/target/{}.wav".format(str(uuid.uuid4()))
    # os.rename(audio.name, src_path)
    (ffmpeg.input(audio)
            .output(src_path, acodec='pcm_s16le', ac=1, ar=22050)
            .run()
    )
    
    model.predict(src_path, tgt_path)
    return tgt_path


inputs = gr.inputs.Audio(label="Source Audio", source="microphone", type='filepath')
outputs = gr.outputs.Audio(label="Target Audio", type='filepath')

title = "Chinese-to-English Direct Speech-to-Speech Translation (BETA)"
#"""
gr.Interface(
    denoising_transform, inputs, outputs, title=title,
    allow_flagging='never',
).launch(
    server_name='localhost',
    server_port=7871,
    #ssl_keyfile='example.key',
    #ssl_certfile="example.crt",
)
