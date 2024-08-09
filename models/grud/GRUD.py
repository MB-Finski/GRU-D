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

from models.grud.GRUDCell import GRUDCell

class OutputDropout(nn.Module):
    def __init__(self, dropout: float = 0.1, device: str ='cpu'):
        super(OutputDropout, self).__init__()
        
        self.dropout = dropout
        self.mask = None
        self.device = device
        
    def reset_dropout_mask(self, batch_size, hidden_size):
        self.mask = torch.bernoulli(torch.ones(batch_size, hidden_size) * (1 - self.dropout)).to(self.device)
        
    def forward(self, input):
        # Input is of shape (batch, hidden_size)
        if self.training:
            return input * self.mask
        else:
            return input

class GRUD(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, x_mean: torch.Tensor | None = None, 
                 dropout: None | float = None, device: str ='cpu', bias: bool =True, dropout_type: str ='mloss',
                 feed_missing_mask: bool = True, output_dropout: float | None = 0.0):   
        super(GRUD, self).__init__()
        
        # Validate the inputs
        if num_layers < 1:
            raise ValueError("num_layers should be greater than 0")
        if hidden_size < 1:
            raise ValueError("hidden_size should be greater than 0")
        if input_size < 1:
            raise ValueError("input_size should be greater than 0")
        if x_mean is not None and x_mean.size(-1) != input_size:
            raise ValueError("x_mean should have the same number of features as input_size")
        if dropout is not None and (dropout > 1.0 or dropout < 0.0):
            raise ValueError("dropout should be a number in range [0, 1] representing the probability of an element being zeroed")
        if dropout_type not in ['mloss', 'gal']:
            raise ValueError("dropout_type should be either 'mloss' or 'gal'")
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.x_mean = x_mean
        self.input_size = input_size
        self.feed_missing_mask = feed_missing_mask
        self.dropout_type = dropout_type
        
        self.grud_cells = nn.ModuleList()
        
        if output_dropout is not None and dropout_type == 'gal':
            self.output_dropouts = []
            for _ in range(num_layers):
                self.output_dropouts.append(OutputDropout(dropout, device))            
        else:            
            self.output_dropouts = None
        
        # First GRUDCell
        self.grud_cells.append(GRUDCell(input_size=input_size, hidden_size=hidden_size, x_mean=x_mean, 
                                        dropout=dropout, device=device, bias=bias, dropout_type=dropout_type,
                                        feed_missing_mask=feed_missing_mask))
        
        # Additional GRUDCells
        for _ in range(num_layers-1):
            self.grud_cells.append(GRUDCell(input_size=hidden_size, hidden_size=hidden_size, mask_size=input_size,
                                            delta_time_size=input_size, x_mean=None, dropout=dropout, device=device, bias=bias,
                                            dropout_type=dropout_type, feed_missing_mask=feed_missing_mask))
            
    def initialize_hidden(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.hidden_size, device=device)
    

        
    def forward(self, input):
        # Input should be of shape (batch, 3, seq_len, features)
        
        batch_size = input.size(0)
        seq_len = input.size(2)
        
        if self.dropout_type == 'gal':
            
            # Reset the dropout masks for W, U and V before each batch
            # Such that the same mask is applied for each time step
            for cell in self.grud_cells:
                cell.reset_dropout_mask(batch_size)
        
        
        #x_latest_observations = torch.zeros(batch_size, self.input_size).to(input.device)
        if self.feed_missing_mask:
            x_latest_observations = self.x_mean.repeat(batch_size, 1).to(input.device)
        else:
            # TODO: we're just assuming that 1 is less likely to be a real observation
            # This is a bit of a hack, but it works as long as the input is standardized
            #x_latest_observations = torch.ones(batch_size, self.input_size).to(input.device) * -5.0
            x_latest_observations = torch.ones(batch_size, self.input_size).to(input.device)
            x_has_been_observed = torch.zeros(batch_size, self.input_size).to(input.device)
    
        hidden_states = []
        last_layer_hidden_states = torch.empty(seq_len, batch_size, self.hidden_size).to(input.device)
        
        # Initialize hidden states and reset dropout masks for the output dropout
        for i in range(self.num_layers):
            hidden_states.append(self.initialize_hidden(input.size(0)))
            
            if self.output_dropouts is not None:
                self.output_dropouts[i].reset_dropout_mask(batch_size, self.hidden_size)
        
        
        # Step through the sequence
        for sequence_index in range(seq_len):
            step_input = input[:,:,sequence_index,:].squeeze()
            step_mask = step_input[:,1,:].squeeze()
            step_x = step_input[:,0,:].squeeze()
            
            for i, cell in enumerate(self.grud_cells):
                
                if i == 0:
                    # First layer
                    # Only the first layer has use for x_latest_observations
                    hidden_state = cell(step_input, hidden_states[i], x_latest_observations = x_latest_observations,
                                        x_has_been_observed = x_has_been_observed if not self.feed_missing_mask else None)
                else:
                    hidden_state = cell(step_input, hidden_states[i], x_latest_observations = None, x_has_been_observed = None)
                    
                if self.output_dropouts is not None: # Only applies to gal dropout
                    hidden_state = self.output_dropouts[i](hidden_state)          
                
                hidden_states[i] = hidden_state
                
            # Update x_latest_observations and save the hidden state of the last layer for this time point
            x_latest_observations = torch.where(step_mask > 0, step_x, x_latest_observations)
            if not self.feed_missing_mask:
                x_has_been_observed = torch.where(step_mask > 0, torch.ones_like(x_has_been_observed).to(input.device), x_has_been_observed)
            last_layer_hidden_states[sequence_index] = hidden_state
            
        return last_layer_hidden_states
                
            
            
       

    