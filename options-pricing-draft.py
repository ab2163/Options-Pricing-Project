#Imports
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from py_vollib.black_scholes import black_scholes
#from py_vollib.black_scholes.implied_volatility import implied_volatility
#from py_vollib.black_scholes.greeks import analytical
from random import uniform
from sklearn.preprocessing import StandardScaler

print('Imports complete')

#Option parameters
num_samples = 2100
test_ratio = 1/7
spots = [uniform(10, 500) for p in range(0, num_samples)]
strikes = [uniform(0.6*spots[p], 1.1*spots[p]) for p in range(0, num_samples)]
mat_times = [uniform(0.01, 2) for p in range(0, num_samples)]
vols = [uniform(0.1, 0.2) for p in range(0, num_samples)]
int_rates = [uniform(0.05, 0.20) for p in range(0, num_samples)]

#Black-Scholes option prices
call_prices = []
for i in range(0, num_samples):
	call_prices.append(black_scholes('c', spots[i], strikes[i], mat_times[i], int_rates[i], vols[i]))

print('Data generation complete')

#Normalize data
scaler = StandardScaler()
spots = scaler.fit_transform(spots)
strikes = scaler.fit_transform(srtikes)
mat_times = scaler.fit_transform(mat_times)
vols = scaler.fit_transform(vols)
int_rates = scaler.fit_transform(int_rates)

print('Data normalisation complete')

class OptionsDataset(Dataset):
    def __init__(self, training_set):
        if training_set:
            self.len_data = int((1 - test_rato)*num_samples)
            self.offset = 0
        else:
            self.len_data = int(test_rato*num_samples)
            self.offset = int((1 - test_rato)*num_samples)

    def __len__(self):
        return self.len_data

    def __getitem__(self, i):
        i = i + self.offset
        input_vals = torch.tensor([spots[i], strikes[i], mat_times[i], int_rates[i], vols[i]])
        label = torch.tensor(call_prices[i])
        return input_vals, label

#Neural network definition
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
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

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#Main script
training_data = OptionsDataset(true)
test_data = OptionsDataset(false)
train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)
model = NeuralNetwork()
learning_rate = 1e-3
batch_size = 64
epochs = 1
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")