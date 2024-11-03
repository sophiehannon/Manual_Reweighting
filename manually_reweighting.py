# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 09:57:14 2024

@author: sophi
"""

import mplhep
import numpy as np
import pandas as pd
import uproot
from matplotlib import pyplot as plt
import os
import pickle

import time

start_time = time.time()

## Prints time program is started to help keep track
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("The program started at :", current_time)


pickle_dir = 'Pickles'

home = os.getcwd()

def unpickle_data2(pickle_dir, data_dict):
    os.chdir(home)
    os.chdir(pickle_dir)
    # access a list of all files in pickled data
    all_entries = os.listdir()
    # Create empty arrays to put unpickeled data into
    list_theta_x = [0]*len(all_entries)
    list_theta_y = [0]*len(all_entries)
    list_k = [0]*len(all_entries)
    for file_index in range(len(all_entries)):
        with open(all_entries[file_index], "rb") as file: # Read in pickles and assign to lists
            loaded_object = pickle.load(file)
        list_theta_x[file_index] = loaded_object[0]
        list_theta_y[file_index] = loaded_object[1]
        list_k[file_index] = loaded_object[2]
        data_dict = {
            'theta_x' : list_theta_x,
            'theta_y' : list_theta_y,
            'k' : list_k,
        }
    return data_dict

def unpickle_data(pickle_file, data_dict):
    os.chdir(home)
    os.chdir(pickle_dir)
    # access a list of all files in pickled data
    all_entries = os.listdir()
    # Create empty arrays to put unpickeled data into
    #list_theta_x = [0]*len(all_entries)
    #list_theta_y = [0]*len(all_entries)
    #list_k = [0]*len(all_entries)
    #for file_index in range(len(all_entries)):
    with open(pickle_file, "rb") as file: # Read in pickles and assign to lists
        loaded_object = pickle.load(file)
    list_theta_x = loaded_object[0]
    list_theta_y = loaded_object[1]
    list_k = loaded_object[2]
    data_dict = {
        'theta_x' : list_theta_x,
        'theta_y' : list_theta_y,
        'k' : list_k,
    }
    return data_dict

def calculate_weights(d0, d0b):
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.divide(d0, d0b)
        weights[d0b == 0] = 1  # Set weight to 1 where both histograms are empty
    return weights

def calculate_pull(h_d, h_db):
    with np.errstate(divide='ignore', invalid='ignore'):
        pulls = np.divide((h_d - h_db), np.sqrt(h_d))
        pulls[np.sqrt(h_d) == 0] = 0  # Set pull to 0 where both histograms are empty
    return pulls

def iterate_weights(n, d0_data, d0b_data):
    k_weights = np.ones(40)
    i=0
    while i < n:
        # Create x histograms
        h_x_d, bin_edges_x_d = np.histogram(d0_data['theta_x'], bins=40, range=[-0.3,0.3])
        h_x_db, bin_edges_x_db = np.histogram(d0b_data['theta_x'], bins=40, range=[-0.3,0.3])
        # Apply weights to x
        h_x_db_w = k_weights * h_x_db
        #Calculate 1D weights
        x_weights = calculate_weights(h_x_d, h_x_db_w)
        #Calculate y histograms
        h_y_d, bin_edges_y_d = np.histogram(d0_data['theta_y'], bins=40, range=[-0.3,0.3])
        h_y_db, bin_edges_y_db = np.histogram(d0b_data['theta_y'], bins=40, range=[-0.3,0.3])
        # Apply x weights to y
        h_y_db_w = h_y_db * x_weights
        # Calculate y weights
        y_weights = calculate_weights(h_y_d, h_y_db_w)
        #Create k histograms    
        h_k_d, bin_edges_k_d = np.histogram(d0_data['k'], bins=40, range=[0.0, 7.5e-4])
        h_k_db, bin_edges_k_db = np.histogram(d0b_data['k'], bins=40, range=[0.0, 7.5e-4])
        # Apply y weights to k histograms
        h_k_db_w = h_k_db * y_weights
        k_weights = calculate_weights(h_k_d, h_k_db_w)
        i +=1
    hists = [h_x_d, h_x_db_w, h_y_d, h_y_db_w, h_k_d, h_k_db_w]
    bin_edges = [bin_edges_x_d, bin_edges_x_db, bin_edges_y_d, bin_edges_y_db, bin_edges_k_d, bin_edges_k_db]
    weights = np.array([x_weights, y_weights, k_weights])
    reshaped_weights = weights.reshape(40,3)
    return hists, bin_edges, reshaped_weights


# Dictionaries to unpickle into
d_0 = {}
d_0_b = {}

d0 = unpickle_data('d0.pickle', d_0)
d0b = unpickle_data('db.pickle', d_0_b)

print(len(d0['theta_x']))
print(len(d0b['theta_x']))

n = 1

x_pull = [0,10]

while n < 1e4:

    print(max(x_pull))
        
    hists, bin_edges, weights = iterate_weights(n, d0, d0b)
    h_x_d_w, h_x_db_w, h_y_d_w, h_y_db_w, h_k_d_w, h_k_db_w = hists
    bin_edges_x_d, bin_edges_x_db, bin_edges_y_d, bin_edges_y_db, bin_edges_k_d, bin_edges_k_db = bin_edges
    

    # weighted pulls
    x_pull = calculate_pull(h_x_d_w, h_x_db_w)
    y_pull = calculate_pull(h_y_d_w, h_y_db_w)
    k_pull = calculate_pull(h_k_d_w, h_k_db_w)
    
    
    print(f'\n\nthe n value is {n}\n')
    print(f'max x pull = {max(x_pull)}')
    print(f'min x pull = {min(x_pull)}')
    print(f'max y pull = {max(y_pull)}')
    print(f'min  ypull = {min(y_pull)}')
    print(f'max k pull = {max(k_pull)}')
    print(f'min k pull = {min(k_pull)}')
    
    
    fig, ax = plt.subplots(2, 3, figsize=(30, 10))
    plt.suptitle(r'$\theta_{x} , \theta_{y}$ and k histograms for 144.813 MeV < $\Delta$m < 146.066 MeV', fontsize = 25)
    
    #Plot histograms
    mplhep.histplot(h_x_d_w, bin_edges_x_d, ax=ax[0,0], label=r'D$^{0}$', histtype='fill', color='skyblue')
    mplhep.histplot(h_x_db_w, bin_edges_x_db, ax=ax[0,0], label=r'$\bar{D}^{0}$', histtype='step', hatch='//', facecolor='none', edgecolor='red') 
    
    mplhep.histplot(h_y_d_w, bin_edges_y_d, ax=ax[0,1], label=r'D$^{0}$',  histtype='fill', color='skyblue') 
    mplhep.histplot(h_y_db_w, bin_edges_y_db, ax=ax[0,1], label=r'$\bar{D}^{0}$', histtype='step', hatch='//', facecolor='none', edgecolor='red') 
    
    mplhep.histplot(h_k_d_w, bin_edges_k_d, ax=ax[0,2], label=r'D$^{0}$',  histtype='fill', color='skyblue') 
    mplhep.histplot(h_k_db_w, bin_edges_k_db, ax=ax[0,2], label=r'$\bar{D}^{0}$', histtype='step', hatch='//', facecolor='none', edgecolor='red') 
    
    # Label axes
    ax[0,0].set_xlabel(r'$\theta$ x')
    ax[1,0].set_xlabel(r'$\theta$ x')
    ax[1,0].set_ylabel('Pull')
    ax[0,1].set_xlabel(r'$\theta$ y')
    ax[1,1].set_xlabel(r'$\theta$ y')
    ax[1,1].set_ylabel('Pull')
    ax[0,2].set_xlabel('k')
    ax[1,2].set_xlabel('k')
    ax[1,2].set_ylabel('Pull')
    
    # Plot asymmetries
    ax[1,0].scatter(np.linspace(-0.3,0.3,len(x_pull)), x_pull, color='k', marker='x')
    ax[1,1].scatter(np.linspace(-0.3,0.3,len(y_pull)), y_pull, color='k', marker='x')
    ax[1,2].scatter(np.linspace(0.0, 7.5e-4,len(k_pull)), k_pull, color='k', marker='x')
    
    # Plot line of zeros
    ax[1,0].plot(np.linspace(-0.3,0.3,len(x_pull)), np.zeros(len(x_pull)), color='r', linestyle='--')
    ax[1,1].plot(np.linspace(-0.3,0.3,len(y_pull)), np.zeros(len(y_pull)), color='r', linestyle='--')
    ax[1,2].plot(np.linspace(0,7.5e-4,len(k_pull)), np.zeros(len(k_pull)), color='r', linestyle='--')
    
    # Adjust layout
    plt.tight_layout(pad=2.0)  # Add padding between subplots to avoid overlap
    
    for i in [0,1,2]:
        ax[0,i].set_ylabel('Counts')
        ax[0,i].legend()
    
    os.chdir(home)
    if (os.path.isdir('Manually weighting') == False):
        os.mkdir('Manually weighting')
    os.chdir('Manually weighting')
    if (os.path.isdir('Plots') == False):
        os.mkdir('Plots')
    os.chdir('Plots')
    plt.savefig(f"iteration_{n}.png", dpi = 500)
    plt.show()
    os.chdir('..')
    if (os.path.isdir('Weights') == False):
        os.mkdir('Weights')
    os.chdir('Weights')
    np.savetxt(f"weights_iteration_{n}.csv", weights, delimiter=",")
    os.chdir(home)
    n = 10*n
    print(n)
