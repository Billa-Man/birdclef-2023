import os
import math
import matplotlib.pyplot as plt

import torch
import torchaudio


# Create spectrogram directly from audio

def create_spectrogram_direct(audio_filepath, output_path):

    audio, sr = torchaudio.load(audio_filepath)
    audio_spectrogram = torchaudio.transforms.Spectrogram()(audio)
    print(audio.size(0), audio.size(1))

    duration = audio.size(1) * 1e-5
    height = 3

    plt.figure(figsize=(duration, height))
    plt.axis("off")
    plt.imshow(audio_spectrogram.log2()[0, :, :].numpy(), cmap='viridis', aspect='auto', origin='lower')
    plt.savefig(output_path, dpi=100, format='png', bbox_inches='tight', transparent=True)
    plt.close()


## WINDOW METHOD
    
# Create spectrogram of split audio samples

def create_spectrogram(audio, output_path):

    audio_spectrogram = torchaudio.transforms.Spectrogram()(audio)

    duration = audio.size(1) / audio.size(0)
    height = 3
    width = duration * 4

    plt.figure(figsize=(width, height))
    plt.axis("off")
    plt.imshow(audio_spectrogram.log2()[0, :, :].numpy(), cmap='viridis', aspect='auto', origin='lower')
    plt.savefig(output_path, dpi=200, format='png', bbox_inches='tight', transparent=True)
    plt.close()


# Process audio samples (can be of varying lengths)

def create_audio_samples(audio_filepath, output_path, window=30, overlap=10):

    waveform, sample_rate = torchaudio.load(audio_filepath)

    window_samples = int(window*sample_rate)
    overlap_samples = int(overlap*sample_rate)
    total_samples = waveform.size(1)
    num_segments = math.ceil(total_samples / (window_samples - overlap_samples))
    
    os.makedirs(output_path, exist_ok=True)

    for i in range(num_segments):

        start_pos = i * (window_samples-overlap_samples)
        end_pos = start_pos + window_samples

        if (end_pos < (total_samples)):
            segment = waveform[:, start_pos:end_pos]
            create_spectrogram(segment, os.path.join(output_path, f"segment_{i + 1}.png"))
        else: 
            final_part = waveform[:, start_pos:]
            silence = torch.zeros((waveform.size(0), window_samples - final_part.size(1)))
            final_part = torch.cat([final_part, silence], dim=1)
            create_spectrogram(final_part, os.path.join(output_path, f"segment_{i + 1}.png"))
            print("YES")