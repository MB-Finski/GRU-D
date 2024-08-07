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
from sklearn.metrics import roc_curve, auc, roc_auc_score

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def fit(model, criterion, learning_rate,
        train_dataloader, test_dataloader,
        n_epochs=30, device='cpu', weight_decay=0, patience=10, min_delta=0.001):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    early_stopping = EarlyStopper(patience=patience, min_delta=min_delta)
    
    best_testing_loss = float('inf')
    stored_auc = 0
    
    for epoch in range(n_epochs):
        
        # Train the model
        losses= []
        model.train()
        n=0
        for train_data, train_label, train_last_obs in train_dataloader:
            optimizer.zero_grad()
            
            train_data, train_label, train_last_obs = train_data.to(device), train_label.to(device), train_last_obs.to(device)
            
            # Forward pass
            y_pred = model(train_data, train_last_obs)
            
            # Compute loss
            loss = criterion(y_pred.squeeze(), train_label.squeeze())
            losses.append(loss.item())

            # Backward pass
            loss.backward()
            
            # Update the parameters
            optimizer.step()
            

        train_loss = np.mean(losses)
        
        # Test the model
        losses = []        
        model.eval()
        label, pred = [], []        
        for test_data, test_label, test_last_obs in test_dataloader:
            test_data, test_label, test_last_obs = test_data.to(device), test_label.to(device), test_last_obs.to(device)
            
            # Forward pass
            y_pred = model(test_data, test_last_obs)
            
            # Compute loss
            loss = criterion(y_pred.squeeze(), test_label.squeeze())
            losses.append(loss.item())
            
            # Store the predictions and correct labels
            label = np.append(label, test_label.detach().cpu().numpy())
            pred = np.append(pred, y_pred.detach().cpu().numpy())
        
        # Apply sigmoid to the prediction
        pred = 1/(1+np.exp(-pred))
        
        test_loss = np.mean(losses)
        
        auc_score = roc_auc_score(label, pred)
        
        if test_loss < best_testing_loss:
            best_testing_loss = test_loss
            stored_auc = auc_score
        
        print("Epoch: {} Train loss: {:.4f}, Test loss: {:.4f}, Test AUC: {:.4f}".format(epoch, train_loss, test_loss, auc_score))
        
        #  Early stopping
        if early_stopping.early_stop(test_loss):
            print("Early stopping")
            break
    
    return stored_auc
        