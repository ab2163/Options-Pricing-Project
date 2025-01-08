#nohup ~/BTS-Work/bin/python3 options-pricing-draft.py > training_log 2>&1 &

#Imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from py_vollib.black_scholes import black_scholes
from random import uniform
import numpy as np
import pickle
import sys

#Option parameters
num_samples = 35000
test_ratio = 1/7
spots = np.array([uniform(10, 500) for p in range(0, num_samples)])
strikes = np.array([uniform(0.6*spots[p], 1.1*spots[p]) for p in range(0, num_samples)])
mat_times = np.array([uniform(0.01, 2) for p in range(0, num_samples)])
vols = np.array([uniform(0.1, 0.2) for p in range(0, num_samples)])
int_rates = np.array([uniform(0.05, 0.20) for p in range(0, num_samples)])

#Calculate Black-Scholes option prices
call_prices = []
for i in range(0, num_samples):
	call_prices.append(black_scholes('c', spots[i], strikes[i], mat_times[i], int_rates[i], vols[i]))

#Calculate means
spots_mean = spots.mean()
strikes_mean = strikes.mean()
mat_times_mean = mat_times.mean()
vols_mean = vols.mean()
int_rates_mean = int_rates.mean()

#Calculate standard deviations
spots_std = spots.std()
strikes_std = strikes.std()
mat_times_std = mat_times.std()
vols_std = vols.std()
int_rates_std = int_rates.std()

#Normalize data
spots = (spots - spots_mean)/spots_std
strikes = (strikes - strikes_mean)/strikes_std
mat_times = (mat_times - mat_times_mean)/mat_times_std
vols = (vols - vols_mean)/vols_std
int_rates = (int_rates - int_rates_mean)/int_rates_std

#Save dataset
dataset_out = {}
dataset_out['spots'] = spots
dataset_out['strikes'] = strikes
dataset_out['mat_times'] = mat_times
dataset_out['vols'] = vols
dataset_out['int_rates'] = int_rates
dataset_out['spots_mean'] = spots_mean
dataset_out['strikes_mean'] = strikes_mean
dataset_out['mat_times_mean'] = mat_times_mean
dataset_out['vols_mean'] = vols_mean
dataset_out['int_rates_mean'] = int_rates_mean
dataset_out['spots_std'] = spots_std
dataset_out['strikes_std'] = strikes_std
dataset_out['mat_times_std'] = mat_times_std
dataset_out['vols_std'] = vols_std
dataset_out['int_rates_std'] = int_rates_std
with open('Black-Sch-Data.pkl', 'wb') as file: 
          pickle.dump(dataset_out, file) 

#Defines a dataset as used by the model training algorithm
class OptionsDataset(Dataset):
    def __init__(self, training_set):
        if training_set:
            self.len_data = int((1 - test_ratio)*num_samples)
            self.offset = 0
        else:
            self.len_data = int(test_ratio*num_samples)
            self.offset = int((1 - test_ratio)*num_samples)

    def __len__(self):
        return self.len_data

    def __getitem__(self, i):
        i = i + self.offset
        input_vals = torch.tensor([spots[i], strikes[i], mat_times[i], int_rates[i], vols[i]])
        label = torch.tensor([call_prices[i]])
        return input_vals, label

#Neural network definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network_stack = nn.Sequential(
            nn.Linear(5, 128),
            nn.Softmax(dim = 1),
            nn.Linear(128, 128),
            nn.Softmax(dim = 1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        logits = self.network_stack(x)
        return logits

#Function which executes one training epoch
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

#Function which evaluates current accuracy of model
def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            #Add up loss function values for each batch
            test_loss += loss_fn(pred, y).item() 
            #Add the number of correct estimations for each batch
            correct += sum([1 if (0.99*yval < mval < 1.01*yval) else 0 for (yval, mval) in zip(y, pred)])

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100*correct

#Setup the model
model = NeuralNetwork()
model.double()
learning_rate = 1e-3
batch_size = 64
epochs = 100000
training_data = OptionsDataset(True)
test_data = OptionsDataset(False)
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model_trg_acc = 99
model_acc = 0

#Train the model
for t in range(epochs):
    train_loop(train_dataloader, model, loss_fn, optimizer)
    if (t+1) % 200 == 0:
        print(f"Epoch {t+1}\n-------------------------------")
        model_acc = test_loop(test_dataloader, model, loss_fn)
        sys.stdout.flush()
        torch.save(model.state_dict(), 'trained_model.pt')
    if model_acc >= model_trg_acc:
         break
print("Finished training")