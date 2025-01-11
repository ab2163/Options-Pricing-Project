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
import QuantLib as ql

#Get filenames from user to export data to
training_set_fname = input('Enter filename (excluding .pkl) to save training data to: ')
model_fname = input('Enter filename (excluding .pt) to save model to: ')

#Function to calculate price of American options
def amer_options_price(spot, strike, mat_time, int_rate, vol):
    #Define Black-Scholes-Merton process
    today = ql.Date().todaysDate()
    riskFreeTS = ql.YieldTermStructureHandle(ql.FlatForward(today, int_rate, ql.Actual365Fixed()))
    dividendTS = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, ql.Actual365Fixed()))
    volatility = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), vol, ql.Actual365Fixed()))
    initialValue = ql.QuoteHandle(ql.SimpleQuote(spot))
    process = ql.BlackScholesMertonProcess(initialValue, dividendTS, riskFreeTS, volatility)

    #Define the option
    option_type = ql.Option.Call
    payoff = ql.PlainVanillaPayoff(option_type, strike)
    end_date = today + int(365*mat_time)
    am_exercise = ql.AmericanExercise(today, end_date)
    american_option = ql.VanillaOption(payoff, am_exercise)

    #Define the pricing engine
    xGrid = 200
    tGrid = 2000
    engine = ql.FdBlackScholesVanillaEngine(process, tGrid, xGrid)
    american_option.setPricingEngine(engine)

    return np.float64(american_option.NPV())

#Option parameters
num_samples = 35000
test_ratio = 1/7
spots = np.array([uniform(10, 500) for p in range(0, num_samples)])
strikes = np.array([uniform(0.6*spots[p], 1.1*spots[p]) for p in range(0, num_samples)])
mat_times = np.array([uniform(0.01, 2) for p in range(0, num_samples)])
vols = np.array([uniform(0.1, 0.2) for p in range(0, num_samples)])
int_rates = np.array([uniform(0.05, 0.20) for p in range(0, num_samples)])
amer_options = True

#Calculate Black-Scholes option prices
call_prices = []
if not amer_options:
    for i in range(0, num_samples):
        call_prices.append(black_scholes('c', spots[i], strikes[i], mat_times[i], int_rates[i], vols[i]))
else:
    for i in range(0, num_samples):
        if i % 1000 == 0 and i > 0:
            print(f'Generating American Option Prices: {i} \n')
        call_prices.append(amer_options_price(spots[i], strikes[i], mat_times[i], int_rates[i], vols[i]))

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
dataset_out['call_prices'] = call_prices
with open(training_set_fname + '.pkl', 'wb') as file: 
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
learning_rate = 1e-4
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
        torch.save(model.state_dict(), model_fname + '.pt')
    if model_acc >= model_trg_acc:
         break
print("Finished training")