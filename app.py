import uuid
import ffmpeg
import gradio as gr
from pathlib import Path
from denoisers.SpectralGating import SpectralGating



def denoising_transform(audio):
    src_path = Path(__file__).parent.resolve() / Path("cache_wav/original/{}.wav".format(str(uuid.uuid4())))
    tgt_path = Path(__file__).parent.resolve() / Path("cache_wav/denoised/{}.wav".format(str(uuid.uuid4())))
    src_path.parent.mkdir(exist_ok=True, parents=True)
    tgt_path.parent.mkdir(exist_ok=True, parents=True)
    (ffmpeg.input(audio)
            .output(src_path.as_posix(), acodec='pcm_s16le', ac=1, ar=22050)
            .run()
    )
    model.predict(audio, tgt_path)
    return tgt_path


demo = gr.Interface(
    fn=denoising_transform,
    inputs=gr.Audio(label="Source Audio", source="microphone", type='filepath'),
    outputs=gr.Audio(label="Target Audio", type='filepath'),
    examples=[
        ["testing/wavs/p232_071.wav"],
        ["testing/wavs/p232_284.wav"],
    ],
    title="Denoising",
    interpretation="default",
)

if __name__ == "__main__":
    model = SpectralGating()
    demo.launch()

