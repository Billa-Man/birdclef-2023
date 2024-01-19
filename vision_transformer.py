import torch
import torch.nn as nn

from functions import create_audio_samples
from config import ModelConfig

# Initialize config file
config = ModelConfig()

# Define model

class VisionTransformer(nn.Module):

    def __init__(self, image_size, patch_size=config.patch_size, num_classes=config.num_classes, 
                 num_channels=config.num_channels, dim_embeddings=config.dim_embeddings,  num_layers=config.num_layers, 
                 num_heads=config.num_heads, mlp_dim=config.mlp_dim, dropout=0.5):
        
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

        final_predictions = []

        for i in range(x.shape[0]):
            spectrogram_samples = create_audio_samples(x[i, :])
            predictions = []
            for j in range(len(spectrogram_samples)):
                predictions.append(self.forward(spectrogram_samples[j]))

            final_predictions.append(torch.mean(torch.stack(predictions), dim=0))
        
        return torch.stack(final_predictions)