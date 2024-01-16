# Import libraries
import matplotlib.pyplot as plt

import os
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

for folder in dirs:
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
LEARNING_RATE = model_parameters.learning_rate
WEIGHT_DECAY = model_parameters.weight_decay
NUM_EPOCHS = model_parameters.num_epochs

model = VisionTransformer(9, 1)
model.to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


# Model training & evaluation
train_loss, val_loss, cMAP_epoch = train_and_eval(model, criterion, optimizer, NUM_EPOCHS, device, train_dataloader, val_dataloader)

# Plot metrics
x_values = range(NUM_EPOCHS)

plt.plot(x_values, train_loss, label='Train Loss')
plt.plot(x_values, val_loss, label='Validation Loss')
plt.plot(x_values, cMAP_epoch, label='cMAP per epoch')
plt.show()

