# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import queue

import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms


class ZoomModel(nn.Module):
    def __init__(self, num_features=10):
        self.num_features = num_features  # From dataframe
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features, hidden_size=64, num_layers=2, batch_first=True
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(64, 64),  # Extract out features
            nn.ReLU(),
            nn.Dropout(0.25),  # prevent overfitting
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Want single value between 1 and 0
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Need to grab output of the last LSTM Stage
        final_timestep = lstm_out[:, -1, :]  # [batch_size, hidden_size]
        out = self.fully_connected(final_timestep)
        return out


""" Functions used for training """


def train_model(model, dataloader, loss_fn, optimizer, device):
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred.squeeze(), y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print("lr: ", optimizer.param_groups[0]["lr"])

    return loss.detach().cpu().numpy()


def test(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            total_loss += loss_fn(pred.squeeze(), y).item()
    return total_loss / len(dataloader)


def infer(model, input_tensor, device):
    model.eval()
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        output = model(input_tensor.unsqueeze(0))  # Add batch dim
        return output.item()


def make_training(labels, features, BATCH_SIZE):

    train_set = [labels, features]

    train_dataloader = DataLoader(
        train_set, batch_size=BATCH_SIZE, drop_last=True, shuffle=True
    )
    test_dataloader = DataLoader(
        train_set, batch_size=BATCH_SIZE, drop_last=False, shuffle=True
    )

    return train_dataloader, test_dataloader


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ZoomModel(num_features=10).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    epochs = 20
    batch_size = 16

    """ Load Labeled Data"""

    features = pd.read_csv("PupilExtraction/output.csv")
    norm_size, loc_x, loc_y = [], [], []
    labels = [norm_size, loc_x, loc_y]

    train_loader, val_loader = make_training(labels, features, batch_size)

    l = []
    v = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        loss = train_model(model, train_loader, loss_fn, optimizer, device)
        validation = test(model, val_loader, loss_fn, device)
        l.append(loss)
        v.append(validation)
        print(f"Train={loss:.4f}, Val={validation:.4f}")

    torch.save(model.state_dict(), "zoom_model.pth")  # Save model

    """ Plot Validation loss curve """

    e = [i for i in range(epochs)]

    min_len = min(len(e), len(l), len(v))

    e = e[:min_len]
    l = l[:min_len]
    v = v[:min_len]

    plt.plot(e, l)
    plt.plot(e, v)
    plt.legend(["Validation", "Loss"])
    plt.title("Validation Loss Curve")
    plt.xlabel("epochs")
    plt.show()

    # Load and infer
    model.load_state_dict(torch.load("zoom_model.pth"))
    sample_input = torch.randn(60, 10)  # Random input
    result = infer(model, sample_input, device)
    print("Zoom Prediction:", result)
