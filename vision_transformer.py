import torch
import torch.nn as nn

patch_size = 16
image_patch_size = 256
num_classes = 10
num_channels = 1
dim_embeddings = 256
num_heads = 6
mlp_dim=1024
num_layers = 2


class VisionTransformer(nn.Module):

    def __init__(self, image_size, patch_size=patch_size, num_classes=num_classes, num_channels=num_channels, dim_embeddings=dim_embeddings, num_layers=num_layers, num_heads=num_heads, mlp_dim=mlp_dim, dropout=0.5):
        super(VisionTransformer, self).__init__()

        self.num_patches_h = image_size[0] // patch_size
        self.num_patches_w = image_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        self.patch_embedding = nn.Conv2d(num_channels, dim_embeddings, kernel_size=patch_size, stride=patch_size)
        self.positional_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim_embeddings))

        self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim_embeddings, num_heads, mlp_dim, dropout), num_layers=num_layers)

        self.classification_layer = nn.Linear(dim_embeddings, num_channels)
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
    
    def average_predictions(self, x, image_patch_size):

        num_patches_h = x.size(2) // image_patch_size
        num_patches_w = x.size(3) // image_patch_size

        predictions = torch.zeros(x.size(0),self.num_classes)

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                patch = x[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]

                patch_prediction = self.forward(patch)
                predictions  = predictions + patch_prediction

        average_predictions = predictions / (num_patches_h*num_patches_w)

        return average_predictions