# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from audio_dataset import AudioDataset
from vision_transformer import VisionTransformer
from config import ModelConfig


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate Dataset
exec(open('generate_dataset.py').read())

dataset_dir = "birdclef-2023/train_waveforms"
dirs = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]
waveform_filepaths = []

for folder in dirs:
    audio_files = [waveform for waveform in os.listdir(os.path.join(dataset_dir, folder))]
    for audio in audio_files:
        waveform_filepaths.append((os.path.join(dataset_dir, folder, audio)))

# Create dataloader
batch_size = 32
audio_dataset = AudioDataset(waveform_filepaths)
data_loader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True)

# Initialise model
model_parameters = ModelConfig()
LEARNING_RATE = model_parameters.learning_rate
WEIGHT_DECAY = model_parameters.weight_decay

model = VisionTransformer(9, 1)
model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Model training & evaluation

