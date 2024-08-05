########################################################################
# Part of a GRU-D implementation and reproduction from the paper:
# https://www.nature.com/articles/s41598-018-24271-9
#
#
# Copyright 2024 Sami Finnilä
########################################################################
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
########################################################################

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.utils.data as utils
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from models.grud.GRUD import GRUD
from classification_fitter import fit

class ClassificationModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, x_mean=0, 
                 bias=True, dropout=0, device='cpu', dropout_type='gal'):
        super(ClassificationModel, self).__init__()
        
        self.grud = GRUD(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, x_mean = x_mean, bias = bias, dropout = dropout, device = device, dropout_type=dropout_type)
        
        fc_layers = []
        #for i in range(2):
        #    fc_layers.append(nn.Linear(hidden_size, hidden_size))
        #    fc_layers.append(nn.ReLU())
        #    fc_layers.append(nn.Dropout(dropout))    
            
        fc_layers.append(nn.Linear(hidden_size, output_size))
        self.fc = nn.Sequential(*fc_layers)
                
        # Initialize weights
        #torch.nn.init.xavier_normal_(self.fc.weight)
        
    def forward(self, input, last_obs):
        x = self.grud(input)
        
        # x.size() = (seq_len, batch_size, hidden_size)
        # We want to get the hidden state at the last time point
        x = torch.gather(x, 0, last_obs.view(1, -1, 1).repeat(1, 1, x.size(-1)).long().to(x.device))
        x = x.squeeze(0)
        
        x = self.fc(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def data_dataloader(dataset, outcomes, last_obs, test_proportion = 0.2, batch_size = 1000):
    
    split_index = int(np.floor(dataset.shape[0] * (1 - test_proportion)))
    
    
    # Split the dataset
    train_data, train_label, train_last_obs = dataset[:split_index, :,:,:], outcomes[:split_index], last_obs[:split_index]
    test_data, test_label, test_last_obs = dataset[split_index:, :,:,:], outcomes[split_index:], last_obs[split_index:]
    
    # Convert to tensor
    train_data, train_label, train_last_obs = torch.Tensor(train_data), torch.Tensor(train_label).unsqueeze(1), torch.Tensor(train_last_obs).unsqueeze(1)
    test_data, test_label, test_last_obs = torch.Tensor(test_data), torch.Tensor(test_label).unsqueeze(1), torch.Tensor(test_last_obs).unsqueeze(1)
    
    # Create datasets
    train_dataset = utils.TensorDataset(train_data, train_label, train_last_obs)
    test_dataset = utils.TensorDataset(test_data, test_label, test_last_obs)
    
    # dataset to dataloader 
    train_dataloader = utils.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=batch_size)
    
    print("Training data shape: ",train_data.shape)
    print("Test data shape: ",test_data.shape)
    
    return train_dataloader, test_dataloader


dataset = np.load('./input/grud_dataset.npy')
outcomes = np.load('./input/grud_outcomes.npy')
last_observation_indices = np.load('./input/grud_last_time_point_indices.npy')

# Suffle the dataset
shuffle_index = np.random.permutation(dataset.shape[0])
dataset = dataset[shuffle_index]
outcomes = outcomes[shuffle_index]
last_observation_indices =last_observation_indices[shuffle_index]

train_dataloader, test_dataloader = data_dataloader(dataset, outcomes, last_observation_indices, test_proportion=0.2, batch_size=1000)

input_size = 33 # Number of features
hidden_size = 49
output_size = 1
num_layers = 1 # Number of GRUD layers

x_mean = torch.Tensor(np.load('./input/grud_x_mean.npy'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

#Options: [gal, mloss]
dropout_type = 'gal' 
model = ClassificationModel(input_size = input_size, hidden_size= hidden_size, output_size=output_size, dropout=0.3, x_mean=x_mean, num_layers=num_layers, bias=True, device=device,dropout_type=dropout_type)

model.to(device)

count = count_parameters(model)
print('number of parameters : ' , count)

# Loss weight for the positive class
#num_positives = np.sum(outcomes)
#num_negatives = len(outcomes) - num_positives
#pos_weight = torch.Tensor([num_negatives/num_positives])
pos_weight = torch.Tensor([1.0])
criterion = torch.nn.BCEWithLogitsLoss(pos_weight = pos_weight.to(device))

learning_rate = 0.005
learning_rate_decay =100 
n_epochs = 500

epoch_losses = fit(model, criterion, learning_rate,
                   train_dataloader, test_dataloader,
                   n_epochs, device=device, weight_decay = 0.001)