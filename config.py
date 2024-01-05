class ModelConfig:
    def __init__(self):

        # ViT Parameters
        self.patch_size = 16
        self.num_classes = 10
        self.num_channels = 1
        self.dim_embeddings = 256
        self.num_heads = 6
        self.mlp_dim=1024
        self.num_layers = 2

        # ViT Hyperparameters
        self.learning_rate = 0.005
        self.weight_decay = 1e-5