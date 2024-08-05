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
import torch.nn.functional as F

class GRUDCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, mask_size: int | None = None, delta_time_size: int | None = None, 
                 x_mean: torch.Tensor | None = None, bias: bool =True, device: str ='cpu', dropout: None | float =0,
                 dropout_type: str ='mloss'):
        super(GRUDCell, self).__init__()
        
        self.device = device
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if mask_size is None:
            mask_size = input_size
        if delta_time_size is None:
            delta_time_size = input_size
        
        if x_mean is None:
            self.first_layer = False
        else:
            self.x_mean = x_mean.clone().detach().requires_grad_(True).to(self.device)
            self.first_layer = True
            # We only need parameters for the diagonal since we want to force the matrix to be diagonal as per the original paper
            self.delta_time_to_gamma_x = nn.Parameter(torch.Tensor(delta_time_size), requires_grad=True)
            self.gamma_x_bias = nn.Parameter(torch.Tensor(delta_time_size), requires_grad=True)
        
        self.dropout_type = 'None'
        if dropout is not None:
            if dropout > 1.0 or dropout < 0.0:
                raise ValueError("dropout should be a number in range [0, 1] representing the probability of an element being zeroed")
            self.dropout = nn.Dropout(dropout)
            self.dropout_type = dropout_type
        
       
        self.delta_time_to_gamma_h = nn.Linear(delta_time_size, hidden_size, bias=bias)
        
        self.reset_gate_lin = nn.Linear(input_size + hidden_size + mask_size, hidden_size , bias=bias)
        self.update_gate_lin = nn.Linear(input_size + hidden_size + mask_size, hidden_size, bias=bias)
        self.new_gate_lin = nn.Linear(input_size + hidden_size + mask_size, hidden_size, bias=bias)
        
        # Initialize weights
        nn.init.xavier_normal_(self.reset_gate_lin.weight)
        nn.init.xavier_normal_(self.update_gate_lin.weight)
        nn.init.xavier_normal_(self.new_gate_lin.weight)
        nn.init.xavier_normal_(self.delta_time_to_gamma_h.weight)
        if self.first_layer:
            # Initialize the parameter vector
            stdv = 1. / (self.delta_time_to_gamma_x.size(-1)**0.5)
            self.delta_time_to_gamma_x.data.uniform_(-stdv, stdv)
        
        # Initialize biases (unnecessary, but included for completeness)
        nn.init.zeros_(self.reset_gate_lin.bias)
        nn.init.zeros_(self.update_gate_lin.bias)
        nn.init.zeros_(self.new_gate_lin.bias)
        nn.init.zeros_(self.delta_time_to_gamma_h.bias)
        if self.first_layer:
            nn.init.zeros_(self.gamma_x_bias)
            
    def reset_dropout_mask(self, batch_size):
        self.dropout_mask = torch.bernoulli(torch.ones(batch_size, self.hidden_size) * (1 - self.dropout.p)).to(self.device)
        
    def forward(self, input, hidden_state, x_latest_observations):
        # Input should be of shape (batch, 3, features)
        # x_latest_observations should be of shape (batch, features)
        X = input[:,0,:].squeeze()
        input_mask = input[:,1,:].squeeze()
        delta_time = input[:,2,:].squeeze()
        
        if self.first_layer:
            # Eq 10:
            w_matrix = torch.diag(self.delta_time_to_gamma_x)
            decay_factor_x = torch.exp(-F.relu(torch.matmul(delta_time, w_matrix)+ self.gamma_x_bias))
            
            # Eq 11:
            X = X * input_mask + (1 - input_mask) * (decay_factor_x * x_latest_observations + (1 - decay_factor_x) * self.x_mean)
            
        # Eq 10:
        decay_factor_h = torch.exp(-F.relu(self.delta_time_to_gamma_h(delta_time)))
        
        # Eq 12:
        hidden_state = decay_factor_h * hidden_state

        
        if self.dropout_type == 'mloss':
            # "Recurrent Dropout without Memory Loss" (https://arxiv.org/abs/1603.05118)
            # Eq 13:
            gate_input = torch.cat([X, hidden_state, input_mask], dim=-1)
            reset_gate = torch.sigmoid(self.reset_gate_lin(gate_input))
            # Eq 14:
            update_gate = torch.sigmoid(self.update_gate_lin(gate_input))
        
            # Eq 15:
            new_state_input = torch.cat([X, reset_gate * hidden_state, input_mask], dim=-1)
            new_state_candidate = torch.tanh(self.new_gate_lin(new_state_input))
            
            new_state_candidate = self.dropout(new_state_candidate)

            # Eq 16:
            hidden_state = (1 - update_gate) * hidden_state + update_gate * new_state_candidate
        if self.dropout_type == 'gal':
            # "A Theoretically Grounded Application of Dropout in Recurrent Neural Networks" (https://arxiv.org/abs/1512.05287)
            hidden_state = hidden_state * self.dropout_mask
            # Eq 13:
            gate_input = torch.cat([X, hidden_state, input_mask], dim=-1)
            reset_gate = torch.sigmoid(self.reset_gate_lin(gate_input))
            # Eq 14:
            update_gate = torch.sigmoid(self.update_gate_lin(gate_input))
        
            # Eq 15:
            new_state_input = torch.cat([X, reset_gate * hidden_state, input_mask], dim=-1)
            new_state_candidate = torch.tanh(self.new_gate_lin(new_state_input))
            
            # Eq 16:
            hidden_state = (1 - update_gate) * hidden_state + update_gate * new_state_candidate
        else:
            # Eq 13:
            gate_input = torch.cat([X, hidden_state, input_mask], dim=-1)
            reset_gate = torch.sigmoid(self.reset_gate_lin(gate_input))
            # Eq 14:
            update_gate = torch.sigmoid(self.update_gate_lin(gate_input))
        
            # Eq 15:
            new_state_input = torch.cat([X, reset_gate * hidden_state, input_mask], dim=-1)
            new_state_candidate = torch.tanh(self.new_gate_lin(new_state_input))
            
            # Eq 16:
            hidden_state = (1 - update_gate) * hidden_state + update_gate * new_state_candidate
        
        return hidden_state