import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import multiprocessing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from audio_dataset import AudioDataset


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
exec(open('generate_dataset.py').read())

dataset_dir = "birdclef-2023/train_waveforms"
audio_dataset = AudioDataset(dataset_dir)

# Create a DataLoader to handle batching
batch_size = 32
data_loader = DataLoader(audio_dataset, batch_size=batch_size, shuffle=True)
