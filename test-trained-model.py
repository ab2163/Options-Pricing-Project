#import os
#os.chdir(r"C:\Users\Ajinkya\Documents\Options-Pricing-Project")
#exec(open('test-trained-model.py').read())

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

#Neural network definition
class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.network_stack = nn.Sequential(
      nn.Linear(5, 128),
      nn.Softmax(dim = 0),
      nn.Linear(128, 128),
      nn.Softmax(dim = 0),
      nn.Linear(128, 1),
    )

  def forward(self, x):
    logits = self.network_stack(x)
    return logits

def loadModel():
  #Load the model
  model = NeuralNetwork()
  model.load_state_dict(torch.load('trained_model.pt', weights_only=True))
  model.eval()
  model.double()
  return model

def loadData():
  #Load in the data
  with open('Black-Sch-Data.pkl', 'rb') as file:
    BS_data = pickle.load(file)
  return BS_data

def euroOptionsPricing(model, ref_data, spot, strike, mat_time, vol, int_rate):

  #Get Black-Scholes predicted value
  pred_eqn = black_scholes('c', spot, strike, mat_time, int_rate, vol)

  #Normalise parameters
  spot = (spot - ref_data['spots_mean'])/ref_data['spots_std']
  strike = (strike - ref_data['strikes_mean'])/ref_data['strikes_std']
  mat_time = (mat_time - ref_data['mat_times_mean'])/ref_data['mat_times_std']
  vol = (vol - ref_data['vols_mean'])/ref_data['vols_std']
  int_rate = (int_rate - ref_data['int_rates_mean'])/ref_data['int_rates_std']
  params = [spot, strike, mat_time, int_rate, vol]
  params = torch.tensor(params)
  
  with torch.no_grad():
    pred = model(params)

  return [pred, pred_eqn]

def testModel():
  model = loadModel()
  BS_data = loadData()
  print(euroOptionsPricing(model, BS_data, 120, 100, 0.5, 0.1, 0.05))
  print(euroOptionsPricing(model, BS_data, 240, 180, 0.25, 0.1, 0.05))
  print(euroOptionsPricing(model, BS_data, 375, 300, 1.0, 0.15, 0.10))
  print(euroOptionsPricing(model, BS_data, 105, 90, 0.1, 0.1, 0.075))
  print(euroOptionsPricing(model, BS_data, 40, 25, 0.3, 0.15, 0.15))
