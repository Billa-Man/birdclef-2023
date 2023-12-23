import os
import math
import matplotlib.pyplot as plt

import torch
import torchaudio


# Create spectrogram of split audio samples

def create_spectrogram(audio, output_path):

    audio_spectrogram = torchaudio.transforms.Spectrogram()(audio)

    plt.figure(figsize=(8, 3))
    plt.axis("off")
    plt.imshow(audio_spectrogram.log2()[0, :, :].numpy(), cmap='viridis')
    plt.savefig(output_path, dpi=200, format='png', bbox_inches='tight', transparent=True)
    plt.close()


# Process audio samples using window method (can be of varying lengths)

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