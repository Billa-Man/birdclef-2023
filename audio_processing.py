import io
import math
import matplotlib.pyplot as plt

import torch
import torchaudio
from torchvision.transforms import ToTensor

from PIL import Image


# Create spectrogram directly from audio split inside the ViT

# def create_spectrogram_direct(audio_filepath):

#     audio, sr = torchaudio.load(audio_filepath)
#     audio_spectrogram = torchaudio.transforms.Spectrogram()(audio)

#     duration = audio.size(1) * 1e-5
#     height = 3

#     plt.figure(figsize=(duration, height))
#     plt.axis("off")
#     plt.imshow(audio_spectrogram.log2()[0, :, :].numpy(), cmap='viridis', aspect='auto', origin='lower')

#     image_stream = io.BytesIO()
#     plt.savefig(image_stream, dpi=100, format='png', bbox_inches='tight', transparent=True)
#     plt.close()

#     image_data = image_stream.getvalue()

#     return image_data


### WINDOW METHOD
    
# Create spectrogram of split audio samples

def create_spectrogram(audio):

    audio_spectrogram = torchaudio.transforms.Spectrogram()(audio)

    duration = audio.size(1) / audio.size(0)
    height = 3
    width = duration * 4

    plt.figure(figsize=(width, height))
    plt.axis("off")
    plt.imshow(audio_spectrogram.log2()[0, :, :].numpy(), cmap='viridis', aspect='auto', origin='lower')

    image_stream = io.BytesIO()
    plt.savefig(image_stream, dpi=100, format='png', bbox_inches='tight', transparent=True)
    plt.close()

    image_data = image_stream.getvalue()
    image_data = Image.open(io.BytesIO(image_data))

    transform = ToTensor()
    image_data = transform(image_data)

    return image_data


# Process audio samples (can be of varying lengths)

def create_audio_samples(audio_filepath, window=30, overlap=10):

    waveform, sample_rate = torchaudio.load(audio_filepath)

    window_samples = int(window*sample_rate)
    overlap_samples = int(overlap*sample_rate)
    total_samples = waveform.size(1)
    num_segments = math.ceil(total_samples / (window_samples - overlap_samples))

    audio_samples = []

    for i in range(num_segments):

        start_pos = i * (window_samples-overlap_samples)
        end_pos = start_pos + window_samples

        if (end_pos < (total_samples)):
            segment = waveform[:, start_pos:end_pos]
            audio_samples.append(create_spectrogram(segment))
        else: 
            final_part = waveform[:, start_pos:]
            silence = torch.zeros((waveform.size(0), window_samples - final_part.size(1)))
            final_part = torch.cat([final_part, silence], dim=1)
            audio_samples.append(create_spectrogram(final_part))

    return audio_samples