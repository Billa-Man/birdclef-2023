# Import libraries
import matplotlib.pyplot as plt

import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from functions import AudioDataset
from vision_transformer import VisionTransformer
from config import ModelConfig
from train_eval import train_and_eval


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate Dataset
exec(open('generate_dataset.py').read())

dataset_dir = "birdclef-2023/train_waveforms"
dirs = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]
waveform_filepaths = []

for folder in tqdm(dirs):
    audio_files = [waveform for waveform in os.listdir(os.path.join(dataset_dir, folder))]
    for audio in audio_files:
        waveform_filepaths.append((os.path.join(dataset_dir, folder, audio)))


# Create dataloader
batch_size = 32
audio_dataset = AudioDataset(waveform_filepaths)

val_size = int(0.2 * len(audio_dataset))
train_size = len(audio_dataset) - val_size

train_dataset, val_dataset = random_split(audio_dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)


# Initialise model
model_parameters = ModelConfig()
model = VisionTransformer(9, 1)
model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=model_parameters.learning_rate, weight_decay=model_parameters.weight_decay)

# Model training & evaluation
train_loss, val_loss, cMAP_epoch = train_and_eval(model, criterion, optimizer, model_parameters.num_epochs, device, train_dataloader, val_dataloader)

# Plot metrics
x_values = range(model_parameters.num_epochs)

plt.plot(x_values, train_loss, label='Train Loss')
plt.plot(x_values, val_loss, label='Validation Loss')
plt.plot(x_values, cMAP_epoch, label='cMAP per epoch')
plt.show()

