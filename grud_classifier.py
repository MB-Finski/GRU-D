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
from classification_fiter import fit
import optuna
import argparse
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description='GRU-D Classifier')
    parser.add_argument('--mode', type=str, default='k_fold', help="Mode: 'k_fold' or 'optim'")
    parser.add_argument('--k', type=int, default=5, help='Number of folds for cross-validation')
    parser.add_argument('--n_trials', type=int, default=200, help='Number of trials for optimization')
    args = parser.parse_args()
    
    if args.mode not in ['k_fold', 'optim', 'compare']:
        raise ValueError("Mode should be either 'k_fold' or 'optim'")    
    if args.mode == 'k_fold' and args.k < 2:
        raise ValueError("Number of folds should be at least 2")    
    if args.mode == 'optim' and args.n_trials < 1:
        raise ValueError("Number of trials should be at least 1")
    
    return args

class ClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, x_mean=0, 
                 bias=True, dropout=0, device='cpu', dropout_type='gal',
                 feed_missing_mask=True):
        super(ClassificationModel, self).__init__()
        
        self.grud = GRUD(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, x_mean = x_mean,
                         bias = bias, dropout = dropout, device = device, dropout_type=dropout_type,
                         feed_missing_mask=feed_missing_mask)
        
        fc_layers = []
        #for i in range(2):
        #    fc_layers.append(nn.Linear(hidden_size, hidden_size))
        #    fc_layers.append(nn.ReLU())
        #    fc_layers.append(nn.Dropout(dropout))    
            
        fc_layers.append(nn.Linear(hidden_size, output_size))
        self.fc = nn.Sequential(*fc_layers)
                
        # Initialize weights
        #nn.init.xavier_normal_(self.fc.weight)
        
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
    
    # Create dataloaders
    train_dataloader = utils.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=0)
    test_dataloader = utils.DataLoader(test_dataset, batch_size=batch_size, num_workers=0)
    
    return train_dataloader, test_dataloader


def do_k_fold_run(k, dropout=0.06, dropout_type='gal', 
                  lr=0.003, weight_decay=0.00001, patience=14,
                  min_delta=-0.005, input_size=33, hidden_size=49,
                  output_size=1, num_layers=1, bias=True, num_epochs=100,
                  feed_missing_mask = True, suffle = True):
    test_proportion = 1/k
    
    criterion = nn.BCEWithLogitsLoss()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the data
    dataset = np.load('./input/grud_dataset.npy')
    outcomes = np.load('./input/grud_outcomes.npy')
    last_time_points = np.load('./input/grud_last_time_point_indices.npy')
    
    x_mean = torch.Tensor(np.load('./input/grud_x_mean.npy'))

    # Shuffle the data
    if suffle:
        shuffle_index = np.random.permutation(dataset.shape[0])
        dataset = dataset[shuffle_index]
        outcomes = outcomes[shuffle_index]
        last_time_points = last_time_points[shuffle_index]
    
    auc_scores = []
    for i in range(k):
        print("-------------------------- Fold: ", i+1, " --------------------------")
        train_dataloader, test_dataloader = data_dataloader(dataset, outcomes, last_time_points, test_proportion = test_proportion, batch_size = 4000)
        
        model = ClassificationModel(input_size = input_size, hidden_size= hidden_size, output_size=output_size, dropout=dropout,
                                    x_mean=x_mean, num_layers=num_layers, bias=bias, device=device,dropout_type=dropout_type,
                                    feed_missing_mask=feed_missing_mask)
        model.to(device)
        
        auc_score = fit(model, criterion, lr, train_dataloader, test_dataloader, n_epochs = num_epochs,
                        device = device, weight_decay = weight_decay, patience=patience, min_delta=min_delta)
        
        auc_scores.append(auc_score)
        
        print("AUC Score: ", auc_score)
        
        # Rotate the data
        dataset = np.roll(dataset, int(dataset.shape[0] * test_proportion), axis=0)
        outcomes = np.roll(outcomes, int(dataset.shape[0] * test_proportion), axis=0)
        last_time_points = np.roll(last_time_points, int(dataset.shape[0] * test_proportion), axis=0)
        
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    
    print("Model parameters: ", count_parameters(model))    
    print("AUC Scores: ", auc_scores)
    print("Mean AUC: ", mean_auc)
    print("Std AUC: ", std_auc)    
        
    return mean_auc

def do_optimization_run(trial):
    input_size = 33 # Number of features
    hidden_size = 49
    output_size = 1
    num_layers = 1 # Number of GRUD layers
    bias = True
    dropout = trial.suggest_float('dropout', 0.0, 0.6)
    dropout_type = 'gal' # 'gal' or 'mloss'
    num_epochs = 50
    lr = trial.suggest_float('lr', 0.00001, 0.01, log=True)
    weight_decay = trial.suggest_float('weight_decay', 0.00001, 0.01, log=True)
    patience = trial.suggest_int('patience', 2, 15)
    min_delta = trial.suggest_float('min_delta', -0.01, 0.01)
    
    auc_score = do_k_fold_run(5, dropout=dropout, dropout_type=dropout_type,
                              lr=lr, weight_decay=weight_decay, patience=patience,
                              min_delta=min_delta, input_size=input_size, hidden_size=hidden_size,
                              output_size=output_size, num_layers=num_layers, bias=bias, num_epochs=num_epochs, suffle=False)
    
    return auc_score

if __name__ == "__main__":
    
    args = parse_args()
    
    if args.mode == 'k_fold':
        k = args.k
        do_k_fold_run(k)
    elif args.mode == 'optim':
        study = optuna.create_study(direction='maximize')
        study.optimize(do_optimization_run, n_trials=args.n_trials)
        
        best_trial = study.best_trial
        print("Best trial:")
        print("Value: ", best_trial.value)
        print("Params: ", best_trial.params)
    elif args.mode == 'compare':
        # Compare two different configurations until one is shown better with 95% confidence
        
        mean_aucs_1 = []
        mean_aucs_2 = []
        
        while True:
            # Configuration 1
            auc_1 = do_k_fold_run(5, feed_missing_mask=True)
            mean_aucs_1.append(auc_1)
            
            # Configuration 2
            auc_2 = do_k_fold_run(5, feed_missing_mask=False, hidden_size=56)
            mean_aucs_2.append(auc_2)
            
            mean_auc_1 = np.mean(mean_aucs_1)
            mean_auc_2 = np.mean(mean_aucs_2)
            std_auc_1 = np.std(mean_aucs_1)
            std_auc_2 = np.std(mean_aucs_2)
            
            print("-----------------------------------------------------------------")
            print("-------------------------Comparison------------------------------")
            print("Mean AUC 1: ", mean_auc_1)
            print("Mean AUC 2: ", mean_auc_2)
            print("Std AUC 1: ", std_auc_1)
            print("Std AUC 2: ", std_auc_2)
            
            if len(mean_aucs_1) < 2:
                continue
            
            # Perform Welch's t-test
            t_stat, p_val = stats.ttest_ind(mean_aucs_1, mean_aucs_2)
            
            print("T-statistic: ", t_stat)
            print("P-value: ", p_val)
            
            if p_val < 0.05:
                if mean_auc_1 > mean_auc_2:
                    print("Configuration 1 is better")
                else:
                    print("Configuration 2 is better")
                break
            
        
        
    