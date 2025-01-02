import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from py_vollib.black_scholes import black_scholes
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes.greeks import analytical
from random import uniform

# Option parameters
num_samples = 100
spots = [uniform(10, 500) for p in range(0, num_samples)]
strikes = [uniform(0.6*spots[p], 1.1*spots[p]) for p in range(0, num_samples)]
mat_times = [uniform(0.01, 2) for p in range(0, num_samples)]
vols = [uniform(0.1, 0.2) for p in range(0, num_samples)]
int_rates = [uniform(0.05, 0.20) for p in range(0, num_samples)]

# Black-Scholes option prices
call_prices = []

for i in range(0, num_samples):
	call_prices.append(black_scholes('c', spots[i], strikes[i], mat_times[i], int_rates[i], vols[i]))
	print(call_prices[i])