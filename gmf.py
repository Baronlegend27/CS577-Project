import torch
from torch import nn


class GMFModel(torch.nn.Module):
    def __init__(self, user_count, item_count, latent_dim_len):
        super(GMFModel, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.latent_dim_len = latent_dim_len
        self.user_embedding = nn.Embedding(num_embeddings=self.user_count, embedding_dim=self.latent_dim_len)
        self.item_embedding = nn.Embedding(num_embeddings=self.item_count, embedding_dim=self.latent_dim_len)

        # Paper suggests Sigmoid activation
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.latent_dim_len, out_features=1),
            nn.Sigmoid())

        self.apply(self.init_weights)

    def forward(self, user_entries, item_entries):
        user_e = self.user_embedding(user_entries)
        item_e = self.item_embedding(item_entries)

        # Matrix product
        product = user_e * item_e

        # Compute output layer score
        score = self.output_layer(product)
        return score.squeeze()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)

