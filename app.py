import uuid
import ffmpeg
import gradio as gr



from denoisers.SpectralGating import SpectralGating

model = SpectralGating()


def denoising_transform(audio):
    src_path = "cache_wav/original/{}.wav".format(str(uuid.uuid4()))
    tgt_path = "cache_wav/denoised/{}.wav".format(str(uuid.uuid4()))
    (ffmpeg.input(audio)
            .output(src_path, acodec='pcm_s16le', ac=1, ar=22050)
            .run()
    )
    model.predict(audio, tgt_path)
    return tgt_path



inputs = gr.inputs.Audio(label="Source Audio", source="microphone", type='filepath')
outputs = gr.outputs.Audio(label="Target Audio", type='filepath')

title = "Denoising"
gr.Interface(
    denoising_transform, inputs, outputs, title=title,
    allow_flagging='never'
).launch(
    # server_name='localhost',
    # server_port=7871
)
