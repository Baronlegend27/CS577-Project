import torch
import os


class ModelEvaluation(object):
    def __init__(self, model, name, device, train_loader, test_loader, lossfn, opt, override=False):
        self.model = model
        self.name = name
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.lossfn = lossfn
        self.opt = opt
        self.override = override

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for user_indices, item_indices, labels in self.train_loader:
                user_indices, item_indices, labels = user_indices.to(self.device), \
                                                     item_indices.to(self.device), \
                                                     labels.to(self.device)

                self.opt.zero_grad()

                # Forward pass
                outputs = self.model(user_indices, item_indices)

                # Compute loss
                loss = self.lossfn(outputs, labels)
                loss.backward()

                self.opt.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1}/{10}, Loss: {avg_loss}")
        torch.save(self.model.state_dict(), f'models/{self.name}.pth')

    def evaluate(self):
        test_loss = 0
        total_samples = 0
        self.model.eval()
        with torch.no_grad():
            for user_indices, item_indices, labels in self.test_loader:
                user_indices, item_indices, labels = user_indices.to(self.device), \
                                                     item_indices.to(self.device), \
                                                     labels.to(self.device)

                # Get the model's predictions
                predictions = self.model(user_indices, item_indices)

                # Calculate the loss for the batch
                loss = self.lossfn(predictions, labels)
                test_loss += loss.item() * len(labels)  # Accumulate weighted loss

                total_samples += len(labels)
        return test_loss / total_samples
