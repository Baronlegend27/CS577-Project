import pandas as pd
import torch.optim
from torch import nn
from evaluation import ModelEvaluation

from gmf import GMFModel
from mlp import MLPModel
from datautils import DataHandler

df = pd.read_csv('data/steam-200k.csv')
df = df.drop(df.columns[-1], axis=1)
handler = DataHandler(df)

mlp = MLPModel(handler.user, handler.items, 8, [64, 16, 4])
gmf = GMFModel(handler.user, handler.items, 8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlp.to(device)
gmf.to(device)

train, test = handler.splitdata()
train_loader, test_loader = handler.dataloaders(train, test)

opt = torch.optim.Adam(mlp.parameters(), lr=0.001)
lossfn = nn.MSELoss()

me_gmf = ModelEvaluation(gmf, 'gmf', "cpu", train_loader, test_loader, lossfn, opt)
me_gmf.train(10)
print(f"Test Loss: {me_gmf.evaluate()}")

me_mlp = ModelEvaluation(mlp, 'mlp', "cpu", train_loader, test_loader, lossfn, opt)
me_mlp.train(10)
print(f"Test Loss: {me_mlp.evaluate()}")
