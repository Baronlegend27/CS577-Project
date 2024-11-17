from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset


class DataHandler(object):
    def __init__(self, data):
        self.data = data
        self.user = len(self.data['UID'].unique())
        self.items = len(self.data['Game'].unique())
        self.time = set(self.data['Playtime'])
        self.playlist = self.data['hasPlayed']
        encoder = LabelEncoder()
        self.data['hasPlayed'] = self.data['hasPlayed'].apply(lambda x: 1.0 if x == 'play' else 0.0)
        self.data['Game'] = encoder.fit_transform(self.data['Game'])
        self.data['UID'] = self.data['UID'].astype("category").cat.codes
        self.data['Playtime'] = self.data['Playtime'] / self.data['Playtime'].max()

    def splitdata(self):
        train, test = train_test_split(self.data, test_size=0.2)  # reproducibility
        return train, test

    def dataloaders(self, train, test):
        train_user_item = torch.tensor(train[['UID', 'Game']].values, dtype=torch.long)
        train_labels = torch.tensor(train['Playtime'].values, dtype=torch.float32)

        test_user_item = torch.tensor(test[['UID', 'Game']].values, dtype=torch.long)
        test_labels = torch.tensor(test['Playtime'].values, dtype=torch.float32)

        train_dataset = TensorDataset(train_user_item[:, 0], train_user_item[:, 1], train_labels)
        test_dataset = TensorDataset(test_user_item[:, 0], test_user_item[:, 1], test_labels)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        return train_loader, test_loader
