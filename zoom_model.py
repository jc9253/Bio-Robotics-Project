%matplotlib inline
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


class zoom_model():
    def __init__(self, batch_size, time_steps, series_shape, num_features=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_features = num_features # From dataframe
        self.batch_size = batch_size

        ## Instantiate Model ##
        super().__init__()
        self.lstm = nn.LSTM(self.num_features, 64, 2, batch_first=True)
        self.fully_connected = nn.Sequential(
            nn.linear(64, 64), #Extract out features
            nn.ReLU(),
            nn.Dropout(0.25), # prevent overfitting
            nn.Linear(64,1),
            nn.Sigmoid() # Want single value between 1 and 0
        )

        self.model = self.to(self.device)

        self.loss_fn = nn.MSELoss() # Output [0, 1]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.5e-3)
        # scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.01, total_iters=100)

    def forward(self, x):
        x = self.lstm(x)
        x = self.fully_connected(x)
        return x

    def make_training(self, BATCH_SIZE):

        self.train_set = 

        self.val_set = 

        self.test_set = 
        
        self.train_dataloader = DataLoader(self.train_set, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
        self.test_dataloader  = DataLoader(self.val_set, batch_size=BATCH_SIZE, drop_last=False, shuffle=True)

        pass

    def train(self, dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
        
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                print("lr: ", optimizer.param_groups[0]['lr'])
        return loss.detach().cpu().numpy()
    
    def test(self, dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return test_loss
    
    def learn(self, epochs, batch_size):
        
        self.make_training(self, batch_size)

        self.epochs = epochs
        self.batch_size = batch_size

        l = []
        v = []

        best_val = 100

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            loss = self.train(self.train_dataloader, self.model, self.loss_fn, self.optimizer)
            validation = self.test(self.test_dataloader, self.model, self.loss_fn)
            l.append(loss)
            v.append(validation)

            # Early stopping:
            if validation < best_val:
                best_val = validation
                epoch_since_best = 0
                torch.save(self.model.state_dict(), "best_model.pth")
            else:
                epoch_since_best += 1
                if epoch_since_best >= 5:
                    print("Validation hit, stopping")
                    break     
                
        print("Done!")

        e = [ i for i in range(epochs)]

        min_len = min(len(e), len(l), len(v))

        e = e[:min_len]
        l = l[:min_len]
        v = v[:min_len]

        # display(l, v, e)

        plt.plot(e, l)
        plt.plot(e, v)
        plt.legend(['Validation', 'Loss'])
        plt.title("Validation Loss Curve")
        plt.xlabel("epochs")
        plt.show()

        torch.save(self.model.state_dict(), "model.pth")


    def infer(self, time_series):
        self.model.eval()
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs) #inference
        return 

if __name__ == "__main__":
    model = zoom_model()
    model.learn()