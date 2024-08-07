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

# A library to load the data from the PhysioNet dataset
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool

# Set a sensible amount of threads for the multiprocessing pool
num_threads = int(np.floor(os.cpu_count() * 0.8))
if num_threads < 1:
    num_threads = 1

init_delta_time_as_ones = False

set_a_file_path = './input/set-a/'

parameter_ids = {'WBC': 0,
                    'RecordID': -1, # Drop
                    'Albumin': 1, 
                    'NISysABP': 2,
                    'NIDiasABP': 3,
                    'NIMAP': 4,
                    'SysABP':2, # Binned together with NISysABP
                    'DiasABP':3, # Binned together with NIDiasABP
                    'MAP': 4, # Binned together with NIMAP
                    'Gender': -1,
                    'Bilirubin': 5,
                    'Temp': 6,
                    'FiO2': 7,
                    'Glucose': 8,
                    'pH': 9,
                    'PaO2': 10,
                    'HCO3': 11,
                    'Weight': 12,
                    'Na': 13,
                    'HCT': 14,
                    'ALP': 15,
                    'Age': -1,
                    'Mg': 16,
                    'TroponinI': 17,
                    'K': 18,
                    'SaO2': 19,
                    'ICUType': -1,
                    'Platelets': 20,
                    'MechVent': -1,
                    'ALT': 21,
                    'Lactate': 22,
                    'Height': -1,
                    'Creatinine': 23,
                    'AST': 24,
                    'GCS': 25,
                    'PaCO2': 26,
                    'TroponinT': 27,
                    'Cholesterol': 28,
                    'RespRate': 29,
                    'Urine': 30,
                    'HR': 31,
                    'BUN': 32}

parameter_filter = [key for key in parameter_ids.keys() if parameter_ids[key] != -1]

parameter_ids = {key: value for key, value in parameter_ids.items() if value != -1}

def physionet_time_to_minutes(time):
    time = time.split(':')
    return int(time[0]) * 60 + int(time[1])

# Load all the files in the set-a directory to memory
print('Loading data to memory...')
data = {}
largest_num_of_unique_time_points = 0
for file in tqdm(os.listdir(set_a_file_path)):
    df = pd.read_csv(set_a_file_path + file)
    id = file.split('.')[0]
    
    # Filter the parameters
    df = df[df['Parameter'].isin(parameter_filter)]
    
    num_unique_time_points = len(df['Time'].unique())
    if num_unique_time_points > largest_num_of_unique_time_points:
        largest_num_of_unique_time_points = num_unique_time_points
    
    # Convert the time to minutes (and int)
    df['Time'] = df['Time'].apply(physionet_time_to_minutes)
    # Ensure the column is int
    df['Time'] = df['Time'].astype(int)
    
    # Convert the Value column to numeric
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    
    data[id] = df
    
# If NI blood pressure and invasive blood pressure are present in the same dataset and during the same time point
# we'll assume that the invasive blood pressure is the correct one and drop the NI blood pressure
print('Merging NISysABP, NIDiasABP and NIMAP with SysABP, DiasABP and MAP...')
def merge_blood_pressure(input):
    df = input[1]
    key = input[0]
    time_points = df['Time'].unique()
    
    for time_point in time_points:
        
        df_time_slice = df[df['Time'] == time_point]
        
        parameters = df_time_slice['Parameter'].values
        
        if 'NISysABP' in parameters:
            if 'SysABP' in parameters:
                df_time_slice = df_time_slice[df_time_slice['Parameter'] != 'NISysABP']
            else:
                df_time_slice = df_time_slice.replace('NISysABP', 'SysABP')
        if 'NIDiasABP' in parameters:
            if 'DiasABP' in parameters:
                df_time_slice = df_time_slice[df_time_slice['Parameter'] != 'NIDiasABP']
            else:
                df_time_slice = df_time_slice.replace('NIDiasABP', 'DiasABP')
        if 'NIMAP' in parameters:
            if 'MAP' in parameters:
                df_time_slice = df_time_slice[df_time_slice['Parameter'] != 'NIMAP']
            else:
                df_time_slice = df_time_slice.replace('NIMAP', 'MAP')
        
        df = df[df['Time'] != time_point]
        df = pd.concat([df, df_time_slice])
    
    return [key, df]

pool = Pool(num_threads)
for ret_val in tqdm(pool.imap_unordered(merge_blood_pressure,data.items()), total=len(data.keys())):    
    data[ret_val[0]] = ret_val[1]
    
# Now we can drop the NISysABP, NIDiasABP and NIMAP from the parameter_ids
parameter_ids = {key: value for key, value in parameter_ids.items() if key not in ['NISysABP', 'NIDiasABP', 'NIMAP']}    

# Find the largest time value in the dataset to determine the size of the time dimension
max_time = 0.0
for key in data.keys():
    if data[key]['Time'].max() > max_time:
        max_time = data[key]['Time'].max()
        
# Find the number of unique parameters in the dataset
parameters = set()
for key in data.keys():
    parameters = parameters.union(set(data[key]['Parameter']))
    
print('Number of unique parameters: ', len(parameters))
print('Max time: ', max_time)
print('Largest number of unique time points: ', largest_num_of_unique_time_points)
print('All parameters: ', parameters)

# Gather all data in one dataframe and use the parameter to create new columns
all_data = pd.concat(data.values())
all_data = all_data.reset_index(drop=True)
all_data = all_data.pivot(columns='Parameter', values='Value')

# Get the mean and sd for each parameter
parameter_means = all_data.mean()
parameter_stdevs = all_data.std()

#Debug:
print('Parameter means: ', parameter_means)
print('Parameter stdevs: ', parameter_stdevs)

# Load the outcomes
outcomes = pd.read_csv('./input/Outcomes-a.txt')
outcomes = outcomes.set_index('RecordID')

# We're only interested in the In-hospital mortality
outcomes = outcomes['In-hospital_death']

# The GRUD model requires the data to be in the shape (observations, 3, time, features)
# The 3 dimensions are for three different matrices: observations, observation_mask and delta_time (since last observation)
# Additionally we need to store a correspoding list of unique time points for each observation


# Create a dictionary to store the data in the correct format
grud_data = {}
num_unique_time_points = {}

print('Creating GRUD-format data...')

def iterate_one_patient(key):
    df = data[key]
    
    # Get the unique parameter_ids values
    number_of_parameters = len(list(set(parameter_ids.values())))
    
    # Create a matrix to store the observations
    observations = np.zeros((largest_num_of_unique_time_points, number_of_parameters))
    observation_mask = np.zeros((largest_num_of_unique_time_points, number_of_parameters))
    delta_time = np.zeros((largest_num_of_unique_time_points, number_of_parameters))
    
    
    # Create a list to store the unique time points
    unique_time_points = df['Time'].unique()
    
    # Ensure that the df is sorted by time
    df = df.sort_values('Time', ascending=True)
    
    time_of_last_observations = {}
    for parameter_id in parameter_ids.values():
        
        if init_delta_time_as_ones:
            time_of_last_observations[parameter_id] = -1.0
        else:
            time_of_last_observations[parameter_id] = 0.0
    
    time_point_index = 0
    # Iterate over all time points
    for time in unique_time_points:
        # Get the observations at the current time point
        obs = df[df['Time'] == time]        
        scaled_time = time / max_time
        
        for _, row in obs.iterrows():
            # Get the index of the parameter
            parameter_index = parameter_ids[row['Parameter']]
            # Store the observation (normalized)
            observations[time_point_index, parameter_index] = (row['Value'] - parameter_means[row['Parameter']]) / parameter_stdevs[row['Parameter']]
            # Store the observation mask
            observation_mask[time_point_index, parameter_index] = 1
            # Store the latest time of observation
            time_of_last_observations[parameter_index] = scaled_time
        
        # Update the delta time matrix
        for parameter_id in parameter_ids.values():
            delta_time[time_point_index, parameter_id] = min(scaled_time - time_of_last_observations[parameter_id], 1.0)
            
        time_point_index += 1
        
    return {'patient_id': key, 'data': np.stack([observations, observation_mask, delta_time], axis=0), 'outcome': outcomes.loc[int(key)], 'last_time_point_indices': len(unique_time_points)-1}

# Iterate over all the patients
patient_keys = list(data.keys())
grud_data_list = []
pool = Pool(num_threads)

for row in tqdm(pool.imap_unordered(iterate_one_patient, patient_keys), total=len(patient_keys)):
    grud_data_list.append(row)


grud_data = []
grud_outcomes = []
last_time_point_indices = []

for row in grud_data_list:
    grud_data.append(row['data'])
    grud_outcomes.append(int(row['outcome']))
    last_time_point_indices.append(row['last_time_point_indices'])
    
print('Number of observations: ', len(grud_data))
print('Number of outcomes: ', len(grud_outcomes))
print('Data shape: ', grud_data[0].shape)

# Get the empirical mean for each parameter in the new grud_data dataset
observations_list = {}
for data_row in grud_data:
    observations = data_row[0]
    data_mask = data_row[1]
    for parameter_id in range(observations.shape[1]):
        if parameter_id not in observations_list:
            observations_list[parameter_id] = []
        for time_point in range(observations.shape[0]):
            if data_mask[time_point, parameter_id] == 1:
                observations_list[parameter_id].append(observations[time_point, parameter_id])


# Double check that the standardization is correct
observations_means = {}
observations_stdevs = {}
for parameter_id in range(observations.shape[1]):
    observations_list[parameter_id] = np.array(observations_list[parameter_id])
    observations_means[parameter_id] = np.mean(observations_list[parameter_id])
    observations_stdevs[parameter_id] = np.std(observations_list[parameter_id])
    

print('Observations means: \n', observations_means)
print('Observations stdevs: \n', observations_stdevs)

# Save the data to disk
np.save('./input/grud_dataset.npy', np.array(grud_data))
np.save('./input/grud_outcomes.npy', np.array(grud_outcomes))
np.save('./input/grud_x_mean.npy', np.array([observations_means[i] for i in range(observations.shape[1])]))
np.save('./input/grud_last_time_point_indices.npy', np.array(last_time_point_indices))
print('Data saved to disk!')