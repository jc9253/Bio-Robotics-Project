import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import queue
import re

import torch
from torch import nn

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision import transforms

from torch.utils.data import TensorDataset
from torch.utils.data import random_split

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class zoom_model(nn.Module):
    def __init__(self, num_features=10):
        self.num_features = num_features  # From dataframe
        hidden_layer = 32
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_layer,
            num_layers=2,
            batch_first=True,
        )
        self.fully_connected = nn.Sequential(
            nn.Linear(hidden_layer, hidden_layer),  # Extract out features
            nn.ReLU(),
            nn.LayerNorm(hidden_layer),
            nn.Dropout(0.5),  # prevent overfitting
            # TODO: THIS IS FOR SINGLE MODEL AT OUTPUT, CHANGE WHEN JACK'S IS READY
            nn.Linear(hidden_layer, 3),
            nn.LayerNorm(3),
            nn.Dropout(0.3),
            nn.Sigmoid(),  # Want single value between 1 and 0, so remember to normalize input to prevent large error
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Need to grab output of the last LSTM Stage
        final_timestep = lstm_out[:, -1]  # , :]  # [batch_size, hidden_size]
        out = self.fully_connected(final_timestep)
        return out


""" Functions used for training """


def train_model(model, dataloader, loss_fn, optimizer, device):
    model.train()

    batch = 0

    for X, y in dataloader:
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
            # print("lv:", loss)
            # print("lv:", current)
            # print("lv:", size)
            print(f"loss: {loss:>7f}  [{current:>5d}]")  # /{size:>5d}]")
            print("lr: ", optimizer.param_groups[0]["lr"])
        batch += 1

    return loss.detach().cpu().numpy()


def test(model, dataloader, loss_fn, device):
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
        return output.squeeze().detach().cpu().numpy()


def make_training(labels, features, BATCH_SIZE):

    # Need as tensors
    # features_tensor = torch.tensor(features, dtype=torch.float32)
    labels_tensor = torch.tensor(np.stack(labels, axis=1), dtype=torch.float32)

    features_tensor = torch.randn(num_samples, sequnce_len, num_features)
    # labels_tensor = torch.randn(num_samples, 3)  # Random input

    dataset = TensorDataset(features_tensor, labels_tensor)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, drop_last=True, shuffle=False
    )
    test_dataloader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, drop_last=False, shuffle=False
    )

    return train_dataloader, test_dataloader


def evaluate_model(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    test_loss = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            all_preds.extend(pred.argmax(1).cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    test_loss /= num_batches
    correct /= size

    print(
        f"Test Set Performance:\n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:.6f}\n"
    )

    return all_preds, all_labels


if __name__ == "__main__":
    num_features = 17
    sequnce_len = 79  # TODO: Will need to update once
    num_samples = 39
    learning_rate = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = zoom_model(num_features=num_features).to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    epochs = 100
    batch_size = 1

    """ Load Labeled Data"""

    df = pd.read_csv("PupilExtraction/features/videos_metrics_formated.csv")

    videos = df["video"].str.replace(r"_[^_]+_face", "", regex=True)
    videos = pd.unique(videos)

    def x_regex(s):
        return re.sub(r"^[^_]+_|_[^_]+_[^_]+", "", s)

    def y_regex(s):
        return re.sub(r"^[^_]+_[^_]+_|_[^_]+", "", s)

    def l_regex(s):
        return re.sub(r"^[^_]+_[^_]+_[^_]+_", "", s)

    x_regex_vec = np.vectorize(x_regex)
    y_regex_vec = np.vectorize(y_regex)
    l_regex_vec = np.vectorize(l_regex)

    loc_x = x_regex_vec(videos)
    loc_y = y_regex_vec(videos)
    size = l_regex_vec(videos)

    loc_x = (loc_x.astype(float)) / 1920
    loc_y = (loc_y.astype(float)) / 1080
    size = (size.astype(float)) / 100

    df1 = df.drop(["video"], axis=1)
    features = torch.from_numpy(df1.values)

    # log = pd.read_csv("data_collection/Braley_log.csv")

    # size = log.loc[:, "Size"]
    # loc_x = log.loc[:, "loc_x"]
    # loc_y = log.loc[:, "loc_y"]
    labels = [size, loc_x, loc_y]

    train_loader, val_loader = make_training(labels, features, batch_size)

    """ Train Model """

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
    print("Model Saved")

    """ Plot Validation loss curve """
    e = [i for i in range(epochs)]

    min_len = min(len(e), len(l), len(v))

    e = e[:min_len]
    l = l[:min_len]
    v = v[:min_len]

    plt.plot(e, l)
    plt.plot(e, v)
    plt.legend(["Loss", "Validation"])
    plt.title("Validation Loss Curve")
    plt.xlabel("epochs")
    plt.show()

    """ Load and infer (Must close plot to see inferece)"""
    print("infereing")
    model.load_state_dict(torch.load("zoom_model.pth"))
    sample_input = torch.randn(sequnce_len, num_features)  # Random input
    result = infer(model, sample_input, device)
    print("Zoom Prediction:", result)

    test_preds, test_labels = evaluate_model(val_loader, model, loss_fn)

    accuracy = accuracy_score(test_labels, test_preds)
    precision = precision_score(
        test_labels, test_preds, average="weighted"
    )  # I mean, shit should be uniform, but weighted seemed like the right move?
    recall = recall_score(test_labels, test_preds, average="weighted")
    f1 = f1_score(test_labels, test_preds, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    cm = confusion_matrix(test_labels, test_preds)

    plt.figure()
    sns.heatmap(
        cm, fmt="d", annot=True
    )  # fuck the way we did this in the hw, this is much easier
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
