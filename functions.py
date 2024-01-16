import io
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchaudio
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

from PIL import Image
from sklearn.metrics import average_precision_score


# Load dataset from filepaths provided

class AudioDataset(Dataset):
    def __init__(self, npy_filepaths):

        self.audio_data = [np.load(filepath) for filepath in npy_filepaths]
        self.audio_data = np.concatenate(self.audio_data)
        self.num_samples = self.audio_data.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.audio_data[idx], dtype=torch.float32)


### WINDOW METHOD
    
# Create spectrogram of split audio samples

def create_spectrogram(waveform):

    audio_spectrogram = torchaudio.transforms.Spectrogram()(waveform)

    duration = waveform.size(1) / waveform.size(0)
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

def create_audio_samples(waveform, sample_rate=32000, window=30, overlap=10):

    window_samples = int(window*sample_rate)
    overlap_samples = int(overlap*sample_rate)
    total_samples = waveform.size(1)
    num_segments = math.ceil(total_samples / (window_samples - overlap_samples))

    spectrogram_samples = []

    for i in range(num_segments):

        start_pos = i * (window_samples-overlap_samples)
        end_pos = start_pos + window_samples

        if (end_pos < (total_samples)):
            segment = waveform[:, start_pos:end_pos]
            spectrogram_samples.append(create_spectrogram(segment))
        else: 
            final_part = waveform[:, start_pos:]
            silence = torch.zeros((waveform.size(0), window_samples - final_part.size(1)))
            final_part = torch.cat([final_part, silence], dim=1)
            spectrogram_samples.append(create_spectrogram(final_part))

    return spectrogram_samples


# Padded cMAP

def cMAP(y_true, y_pred):
    return average_precision_score(y_true, y_pred, average='macro')