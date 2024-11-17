import torch
from torch import nn


class MLPModel(torch.nn.Module):
    def __init__(self, user_count, item_count, genre_count, latent_dim_len, hidden):
        super(MLPModel, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.user_count = user_count
        self.item_count = item_count
        self.genre_count = genre_count  # Number of one-hot encoded genres
        self.latent_dim_len = latent_dim_len
        self.hidden_layer = hidden

        # Embedding layers for user and anime
        self.user_embedding = nn.Embedding(num_embeddings=self.user_count, embedding_dim=self.latent_dim_len)
        self.item_embedding = nn.Embedding(num_embeddings=self.item_count, embedding_dim=self.latent_dim_len)
        self.user_embedding.to(device)
        self.item_embedding.to(device)

        # Define the layers of the MLP
        self.layers = nn.ModuleList()

        # Combine user, item, and non-embedding features
        input_dim = self.latent_dim_len * 2 + 3 + self.genre_count

        # Hidden layers with ReLU activation
        for dim in self.hidden_layer:
            self.layers.append(nn.Linear(in_features=input_dim, out_features=dim))
            self.layers.append(nn.ReLU())
            input_dim = dim

        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=self.hidden_layer[-1], out_features=1),
            nn.ReLU())  # ReLU for output layer can be replaced by sigmoid or identity depending on task

        # Initialize weights
        self.apply(self.init_weights)

    def forward(self, user_entries, item_entries, episodes, name, type, genre):
        # Get user and item embeddings
        user_e = self.user_embedding(user_entries)
        item_e = self.item_embedding(item_entries)

        episodes = episodes.unsqueeze(1)  # (batch_size, 1)
        name = name.unsqueeze(1)  # (batch_size, 1)
        type = type.unsqueeze(1)  # (batch_size, 1)
        # Concatenate all features into one vector
        model_input = torch.cat([user_e, item_e, episodes, name, type, genre], dim=-1)

        # Pass input through hidden layers
        for hidden_layer in self.layers:
            model_input = hidden_layer(model_input)

        # Compute final output score (prediction)
        score = self.output_layer(model_input)
        return score.squeeze()

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
