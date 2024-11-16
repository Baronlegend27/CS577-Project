import torch
from torch import nn


class MLPModel(torch.nn.Module):
    def __init__(self, user_count, item_count, latent_dim_len, hidden):
        super(MLPModel, self).__init__()
        self.user_count = user_count
        self.item_count = item_count
        self.latent_dim_len_len = latent_dim_len
        self.hidden_layer = hidden
        self.user_embedding = nn.Embedding(num_embeddings=self.user_count, embedding_dim=self.latent_dim_len)
        self.item_embedding = nn.Embedding(num_embeddings=self.item_count, embedding_dim=self.latent_dim_len)

        self.layers = nn.ModuleList()

        # Combine user and item embedding dimensions
        input_dim = self.latent_dim_len * 2

        # Paper suggests ReLU activation
        for dim in self.hidden_layer:
            self.layers.append(nn.Linear(in_features=input_dim, out_features=dim))
            self.layers.append(nn.ReLU())
            input_dim = dim

        # Final activation will also be ReLU
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_layer[-1], out_features=1),
            nn.ReLU())

        self.apply(self.init_weights)

    def forward(self, user_entries, item_entries):
        user_e = self.user_embedding(user_entries)
        item_e = self.item_embedding(item_entries)
        model_input = torch.cat([user_e, item_e], dim=-1)

        # Pass input through each layer
        for hidden_layer in self.layers:
            model_input = hidden_layer(model_input)

        # Compute output layer score
        score = self.output_layer(model_input)
        return score.squeeze()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)


