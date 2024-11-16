from animedata import AnimeData
import pandas as pd
from mlp import MLPModel
from gmf import GMFModel
from torch import nn
import torch

df1 = pd.read_csv('data/anime.csv')

df1 = df1
df2 = pd.read_csv('data/rating.csv')
df2 = df2[:25000]

df1.dropna(inplace=True)
df2.dropna(inplace=True)
df2 = df2.loc[df2.iloc[:, 2] != -1]

df1 = df1[df1['episodes'] != 'Unknown']

handler = AnimeData(df2, df1)

user_count = len(handler.ratings['user_id'].unique())  # Number of unique users
item_count = len(handler.anime['anime_id'].unique())  # Number of unique anime
genre_count = 43  # Number of one-hot encoded genre categories
latent_dim_len = 50  # Latent dimension length for embeddings
hidden_layers = [64, 16, 4]  # Example hidden layers for MLP

# Initialize the MLP model
model = MLPModel(user_count=user_count,
                 item_count=item_count,
                 genre_count=genre_count,
                 latent_dim_len=latent_dim_len,
                 hidden=hidden_layers)
a, b = handler.get_loaders()

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for i in range(10):
    model.train()
    total_loss = 0
    for user, anime, type, name, episodes, genre, labels in a:

        # Forward pass
        optimizer.zero_grad()  # Clear previous gradients
        predictions = model(user, anime, episodes, name, type, genre)

        # Calculate loss
        loss = loss_function(predictions, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch [{i + 1}/{10}], Loss: {total_loss / len(a)}")
torch.save(model.state_dict(), f'models/anime.pth')
model.eval()  # Set the model to evaluation mode
total_loss = 0
total_samples = 0

# Loop through the test set
with torch.no_grad():  # Disable gradient calculation
    for user, anime, type, name, episodes, genre, labels in b:
        # Move data to the same device as the model (e.g., GPU)
        user, item = user.to("cpu"), item.to("cpu")
        episodes, name, type, genre = episodes.to("cpu"), name.to("cpu"), type.to("cpu"), genre.to("cpu")
        targets = targets.to("cpu")

        # Get model predictions
        predictions = model(user, item, episodes, name, type, genre)

        # Calculate the loss (e.g., MSE loss)
        loss = loss_function(predictions, labels)
        total_loss += loss.item() * len(targets)  # Accumulate loss
        total_samples += len(targets)  # Accumulate the number of samples

# Calculate average loss
avg_loss = total_loss / total_samples
print(f"Test Loss: {avg_loss:.4f}")
