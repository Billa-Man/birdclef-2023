import torch
import torch.nn as nn

from audio_processing import create_spectrogram
from config import ModelConfig

# Set parameters

config = ModelConfig()

PATCH_SIZE = config.patch_size
NUM_CLASSES = config.num_classes
NUM_CHANNELS = config.num_channels
DIM_EMBEDDINGS = config.dim_embeddings
NUM_HEADS = config.num_heads
MLP_DIM = config.mlp_dim
NUM_LAYERS = config.num_layers


# Define model

class VisionTransformer(nn.Module):

    def __init__(self, image_size, patch_size=PATCH_SIZE, num_classes=NUM_CLASSES, 
                 num_channels=NUM_CHANNELS, dim_embeddings=DIM_EMBEDDINGS,  num_layers=NUM_LAYERS, 
                 num_heads=NUM_HEADS, mlp_dim=MLP_DIM, dropout=0.5):
        
        super(VisionTransformer, self).__init__()

        self.num_patches_h = image_size[0] // patch_size
        self.num_patches_w = image_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.patch_embedding = nn.Conv2d(num_channels, dim_embeddings, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim_embeddings))

        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim_embeddings, num_heads, mlp_dim, dropout), num_layers=num_layers)

        self.classification_layer = nn.Linear(dim_embeddings, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.positional_embedding
        x = self.dropout(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        x = self.classification_layer(x)

        return x

    
    def average_predictions(self, x):

        spectrogram_samples = create_spectrogram(x)

        predictions = torch.zeros(x.size(0),self.num_classes)

        

        for window_x in x:

            patch_prediction = self.forward(window_x)
            predictions  = predictions + patch_prediction

        average_predictions = predictions / ()

        return average_predictions