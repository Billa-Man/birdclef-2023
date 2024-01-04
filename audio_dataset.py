import numpy as np
import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, npy_filepaths):
        
        self.audio_data = [np.load(filepath) for filepath in npy_filepaths]
        self.audio_data = np.concatenate(self.audio_data)
        self.num_samples = self.audio_data.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return torch.tensor(self.audio_data[idx], dtype=torch.float32)