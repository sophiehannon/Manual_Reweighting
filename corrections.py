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


def read_data(files, variables):
    data_frames = []
    total_files = len(files)
    print("\nReading Data...")
 
    for idx, data_file in enumerate(files):
        with uproot.open(data_file) as my_file:
            data_frame = my_file['DecayTree'].arrays(variables, library='pd')
            data_frames.append(data_frame)
            del data_frame  # Delete reference to the temporary DataFrame
        percentage_read = (idx + 1) / total_files * 100
        print(f"Progress: {percentage_read:.2f}%")
 
    print("\nConcatenating Data...")
    total_df = pd.concat(data_frames, ignore_index=True)
    print(f"Total number of events: {len(total_df.index)}")
    return total_df

def split_d_dbar(data_frame):
    d_0 = data_frame.query('piSoft_ID == 211')
    d_0_bar = data_frame.query('piSoft_ID == -211')
    return d_0, d_0_bar

def calculate_theta_x_y_k(data_frame):
    # Remove rows with missing or NaN values in the required columns
    data_frame = data_frame.dropna(subset=['piSoft_D0Fit_PX', 'piSoft_D0Fit_PY', 'piSoft_D0Fit_PZ']).copy()  # Create a copy to avoid the warning
    
    with np.errstate(divide='ignore', invalid='ignore'):  # Suppress warnings for division by zero
        theta_x = np.arctan(data_frame['piSoft_D0Fit_PX'] / data_frame['piSoft_D0Fit_PZ'])
        data_frame.loc[:, 'theta_x'] = theta_x  # Use .loc[] to assign values safely
        
        theta_y = np.arctan(data_frame['piSoft_D0Fit_PY'] / data_frame['piSoft_D0Fit_PZ'])
        data_frame.loc[:, 'theta_y'] = theta_y
        
        k = 1 / np.sqrt(data_frame['piSoft_D0Fit_PX']**2 + data_frame['piSoft_D0Fit_PZ']**2)
        data_frame.loc[:, 'k'] = k
        
    return data_frame


def asymmetry(d0, d0bar):
    return d0 - d0bar


def D_inv_mass(data_frame):
    total_PX_D0Fit = data_frame['H1_D0Fit_PX'] + data_frame['H2_D0Fit_PX'] + data_frame['H3_D0Fit_PX'] + data_frame['H4_D0Fit_PX']
    total_PY_D0Fit = data_frame['H1_D0Fit_PY'] + data_frame['H2_D0Fit_PY'] + data_frame['H3_D0Fit_PY'] + data_frame['H4_D0Fit_PY']
    total_PZ_D0Fit = data_frame['H1_D0Fit_PZ'] + data_frame['H2_D0Fit_PZ'] + data_frame['H3_D0Fit_PZ'] + data_frame['H4_D0Fit_PZ']

    pDotP_D0Fit = total_PX_D0Fit**2 + total_PY_D0Fit**2 + total_PZ_D0Fit**2

    energy_particle_1_D0Fit = np.sqrt(pion_mass**2 + data_frame['H1_D0Fit_PX']**2 + data_frame['H1_D0Fit_PY']**2 + data_frame['H1_D0Fit_PZ']**2)
    energy_particle_2_D0Fit = np.sqrt(pion_mass**2 + data_frame['H2_D0Fit_PX']**2 + data_frame['H2_D0Fit_PY']**2 + data_frame['H2_D0Fit_PZ']**2)
    energy_particle_3_D0Fit = np.sqrt(pion_mass**2 + data_frame['H3_D0Fit_PX']**2 + data_frame['H3_D0Fit_PY']**2 + data_frame['H3_D0Fit_PZ']**2)
    energy_particle_4_D0Fit = np.sqrt(pion_mass**2 + data_frame['H4_D0Fit_PX']**2 + data_frame['H4_D0Fit_PY']**2 + data_frame['H4_D0Fit_PZ']**2)

    energy_D0Fit = energy_particle_1_D0Fit + energy_particle_2_D0Fit + energy_particle_3_D0Fit + energy_particle_4_D0Fit
    
    data_frame['D_inv_M_D0Fit'] = np.sqrt(energy_D0Fit**2 - pDotP_D0Fit)
    
    
    total_PX_ReFit = data_frame['H1_ReFit_PX'] + data_frame['H2_ReFit_PX'] + data_frame['H3_ReFit_PX'] + data_frame['H4_ReFit_PX']
    total_PY_ReFit = data_frame['H1_ReFit_PY'] + data_frame['H2_ReFit_PY'] + data_frame['H3_ReFit_PY'] + data_frame['H4_ReFit_PY']
    total_PZ_ReFit = data_frame['H1_ReFit_PZ'] + data_frame['H2_ReFit_PZ'] + data_frame['H3_ReFit_PZ'] + data_frame['H4_ReFit_PZ']

    pDotP_ReFit = total_PX_ReFit**2 + total_PY_ReFit**2 + total_PZ_ReFit**2

    energy_particle_1_ReFit = np.sqrt(pion_mass**2 + data_frame['H1_ReFit_PX']**2 + data_frame['H1_ReFit_PY']**2 + data_frame['H1_ReFit_PZ']**2)
    energy_particle_2_ReFit = np.sqrt(pion_mass**2 + data_frame['H2_ReFit_PX']**2 + data_frame['H2_ReFit_PY']**2 + data_frame['H2_ReFit_PZ']**2)
    energy_particle_3_ReFit = np.sqrt(pion_mass**2 + data_frame['H3_ReFit_PX']**2 + data_frame['H3_ReFit_PY']**2 + data_frame['H3_ReFit_PZ']**2)
    energy_particle_4_ReFit = np.sqrt(pion_mass**2 + data_frame['H4_ReFit_PX']**2 + data_frame['H4_ReFit_PY']**2 + data_frame['H4_ReFit_PZ']**2)

    energy_ReFit = energy_particle_1_ReFit + energy_particle_2_ReFit + energy_particle_3_ReFit + energy_particle_4_ReFit
    
    data_frame['D_inv_M_ReFit'] = np.sqrt(energy_ReFit**2 - pDotP_ReFit)


def Dst_inv_mass(data_frame):
    total_PX_D0Fit = data_frame['H1_D0Fit_PX'] + data_frame['H2_D0Fit_PX'] + data_frame['H3_D0Fit_PX'] + data_frame['H4_D0Fit_PX'] + data_frame['piSoft_D0Fit_PX']
    total_PY_D0Fit = data_frame['H1_D0Fit_PY'] + data_frame['H2_D0Fit_PY'] + data_frame['H3_D0Fit_PY'] + data_frame['H4_D0Fit_PY'] + data_frame['piSoft_D0Fit_PY']
    total_PZ_D0Fit = data_frame['H1_D0Fit_PZ'] + data_frame['H2_D0Fit_PZ'] + data_frame['H3_D0Fit_PZ'] + data_frame['H4_D0Fit_PZ'] + data_frame['piSoft_D0Fit_PZ']

    pDotP_D0Fit = total_PX_D0Fit**2 + total_PY_D0Fit**2 + total_PZ_D0Fit**2

    energy_particle_1_D0Fit = np.sqrt(pion_mass**2 + data_frame['H1_D0Fit_PX']**2 + data_frame['H1_D0Fit_PY']**2 + data_frame['H1_D0Fit_PZ']**2)
    energy_particle_2_D0Fit = np.sqrt(pion_mass**2 + data_frame['H2_D0Fit_PX']**2 + data_frame['H2_D0Fit_PY']**2 + data_frame['H2_D0Fit_PZ']**2)
    energy_particle_3_D0Fit = np.sqrt(pion_mass**2 + data_frame['H3_D0Fit_PX']**2 + data_frame['H3_D0Fit_PY']**2 + data_frame['H3_D0Fit_PZ']**2)
    energy_particle_4_D0Fit = np.sqrt(pion_mass**2 + data_frame['H4_D0Fit_PX']**2 + data_frame['H4_D0Fit_PY']**2 + data_frame['H4_D0Fit_PZ']**2)
    energy_particle_5_D0Fit = np.sqrt(pion_mass**2 + data_frame['piSoft_D0Fit_PX']**2 + data_frame['piSoft_D0Fit_PY']**2 + data_frame['piSoft_D0Fit_PZ']**2)

    energy_D0Fit = energy_particle_1_D0Fit + energy_particle_2_D0Fit + energy_particle_3_D0Fit + energy_particle_4_D0Fit + energy_particle_5_D0Fit
    
    data_frame['Dst_inv_M_D0Fit'] = np.sqrt(energy_D0Fit**2 - pDotP_D0Fit)
    
    
    total_PX_ReFit = data_frame['H1_ReFit_PX'] + data_frame['H2_ReFit_PX'] + data_frame['H3_ReFit_PX'] + data_frame['H4_ReFit_PX'] + data_frame['piSoft_ReFit_PX']
    total_PY_ReFit = data_frame['H1_ReFit_PY'] + data_frame['H2_ReFit_PY'] + data_frame['H3_ReFit_PY'] + data_frame['H4_ReFit_PY'] + data_frame['piSoft_ReFit_PY']
    total_PZ_ReFit = data_frame['H1_ReFit_PZ'] + data_frame['H2_ReFit_PZ'] + data_frame['H3_ReFit_PZ'] + data_frame['H4_ReFit_PZ'] + data_frame['piSoft_ReFit_PZ']

    pDotP_ReFit = total_PX_ReFit**2 + total_PY_ReFit**2 + total_PZ_ReFit**2

    energy_particle_1_ReFit = np.sqrt(pion_mass**2 + data_frame['H1_ReFit_PX']**2 + data_frame['H1_ReFit_PY']**2 + data_frame['H1_ReFit_PZ']**2)
    energy_particle_2_ReFit = np.sqrt(pion_mass**2 + data_frame['H2_ReFit_PX']**2 + data_frame['H2_ReFit_PY']**2 + data_frame['H2_ReFit_PZ']**2)
    energy_particle_3_ReFit = np.sqrt(pion_mass**2 + data_frame['H3_ReFit_PX']**2 + data_frame['H3_ReFit_PY']**2 + data_frame['H3_ReFit_PZ']**2)
    energy_particle_4_ReFit = np.sqrt(pion_mass**2 + data_frame['H4_ReFit_PX']**2 + data_frame['H4_ReFit_PY']**2 + data_frame['H4_ReFit_PZ']**2)
    energy_particle_5_ReFit = np.sqrt(pion_mass**2 + data_frame['piSoft_ReFit_PX']**2 + data_frame['piSoft_ReFit_PY']**2 + data_frame['piSoft_ReFit_PZ']**2)

    energy_ReFit = energy_particle_1_ReFit + energy_particle_2_ReFit + energy_particle_3_ReFit + energy_particle_4_ReFit + energy_particle_5_ReFit
    
    data_frame['Dst_inv_M_ReFit'] = np.sqrt(energy_ReFit**2 - pDotP_ReFit)


pion_mass = 139.57039

files_2018 = ["../../Data_Files/Preselection/PimPimPipPip_prompt_2018_up_1of4.root", "../../Data_Files/Preselection/PimPimPipPip_prompt_2018_dn_1of4.root",
              "../../Data_Files/Preselection/PimPimPipPip_prompt_2018_up_2of4.root", "../../Data_Files/Preselection/PimPimPipPip_prompt_2018_dn_2of4.root",
              "../../Data_Files/Preselection/PimPimPipPip_prompt_2018_up_3of4.root", "../../Data_Files/Preselection/PimPimPipPip_prompt_2018_dn_3of4.root",
              "../../Data_Files/Preselection/PimPimPipPip_prompt_2018_up_4of4.root", "../../Data_Files/Preselection/PimPimPipPip_prompt_2018_dn_4of4.root"]
 
variables_to_read = ['H1_D0Fit_PX', 'H2_D0Fit_PX', 'H3_D0Fit_PX', 'H4_D0Fit_PX',
                     'H1_D0Fit_PY', 'H2_D0Fit_PY', 'H3_D0Fit_PY', 'H4_D0Fit_PY',
                     'H1_D0Fit_PZ', 'H2_D0Fit_PZ', 'H3_D0Fit_PZ', 'H4_D0Fit_PZ',
                     'H1_ReFit_PX', 'H2_ReFit_PX', 'H3_ReFit_PX', 'H4_ReFit_PX',
                     'H1_ReFit_PY', 'H2_ReFit_PY', 'H3_ReFit_PY', 'H4_ReFit_PY',
                     'H1_ReFit_PZ', 'H2_ReFit_PZ', 'H3_ReFit_PZ', 'H4_ReFit_PZ',
                     'piSoft_ReFit_PE', 'piSoft_ReFit_PX', 'piSoft_ReFit_PY', 'piSoft_ReFit_PZ',
                     'piSoft_D0Fit_PE', 'piSoft_D0Fit_PX', 'piSoft_D0Fit_PY', 'piSoft_D0Fit_PZ',
                     'piSoft_ID']

total_2018 = read_data(files_2018, variables_to_read)

for data_frame in [total_2018]:
    D_inv_mass(data_frame)
    Dst_inv_mass(data_frame)
    data_frame['delta_m_D0Fit'] = data_frame['Dst_inv_M_D0Fit'] - data_frame['D_inv_M_D0Fit']
    data_frame['delta_m_ReFit'] = data_frame['Dst_inv_M_ReFit'] - data_frame['D_inv_M_ReFit']

# Cut into just signal and background
signal_window = total_2018.query('delta_m_ReFit > 144.813 & delta_m_ReFit < 146.066')
total_up_2018_bck_window = total_2018.query('delta_m_ReFit < 144.813 | delta_m_ReFit > 146.066')

time_read = time.time()
print(f"Data read ({(time_read-start_time)/60} mins)")
print(" ")

# Split the data frame into d and dbar
d_0, d_0bar = split_d_dbar(signal_window)

# Calculate theta_x, theta_y and k
d_0 = calculate_theta_x_y_k(d_0)
d_0bar = calculate_theta_x_y_k(d_0bar)

# Create histograms
h_x_d, bin_edges_x_d = np.histogram(d_0['theta_x'], bins=50, range=[-0.3,0.3])
h_x_db, bin_edges_x_db = np.histogram(d_0bar['theta_x'], bins=50, range=[-0.3,0.3])
h_y_d, bin_edges_y_d = np.histogram(d_0['theta_y'], bins=40, range=[-0.3,0.3])
h_y_db, bin_edges_y_db = np.histogram(d_0bar['theta_y'], bins=40, range=[-0.3,0.3])
h_k_d, bin_edges_k_d = np.histogram(d_0['k'], bins=20, range=[0.0, 7.5e-4])
h_k_db, bin_edges_k_db = np.histogram(d_0bar['k'], bins=20, range=[0.0, 7.5e-4])

# Calculate asymmetries
theta_x_asymmetry = (h_x_d - h_x_db) / np.sqrt(h_x_d)
theta_y_asymmetry = (h_y_d - h_y_db) / np.sqrt(h_y_d)
k_asymmetry = (h_k_d - h_k_db) / np.sqrt(h_k_d) 

# Calculate geometric mean distributions
theta_x_mean_geo = np.sqrt(h_x_d * h_x_db)
theta_y_mean_geo = np.sqrt(h_y_d * h_y_db)
k_mean_geo = np.sqrt(h_k_d * h_k_db)

# Calculate arithmetic mean distributions
theta_x_mean_a = (h_x_d + h_x_db) / 2
theta_y_mean_a = (h_y_d + h_y_db) / 2
k_mean_a = (h_k_d + h_k_db) / 2

# Calculate asymmetries between means
theta_x_mean_asymmetry = theta_x_mean_geo - theta_x_mean_a
theta_y_mean_asymmetry = theta_y_mean_geo - theta_y_mean_a
k_mean_asymmetry = k_mean_geo - k_mean_a


fig, ax = plt.subplots(2, 3, figsize=(30, 10))
plt.suptitle(r'$\theta_{x} , \theta_{y}$ and k histograms for 144.813 MeV < $\Delta$m < 146.066 MeV', fontsize = 25)

#Plot histograms
mplhep.histplot(h_x_d, bin_edges_x_d, ax=ax[0,0], label=r'D$^{0}$', histtype='fill', color='skyblue')
mplhep.histplot(h_x_db, bin_edges_x_db, ax=ax[0,0], label=r'$\bar{D}^{0}$', histtype='step', hatch='//', facecolor='none', edgecolor='red') 

mplhep.histplot(h_y_d, bin_edges_y_d, ax=ax[0,1], label=r'D$^{0}$',  histtype='fill', color='skyblue') 
mplhep.histplot(h_y_db, bin_edges_y_db, ax=ax[0,1], label=r'$\bar{D}^{0}$', histtype='step', hatch='//', facecolor='none', edgecolor='red') 

mplhep.histplot(h_k_d, bin_edges_k_d, ax=ax[0,2], label=r'D$^{0}$',  histtype='fill', color='skyblue') 
mplhep.histplot(h_k_db, bin_edges_k_db, ax=ax[0,2], label=r'$\bar{D}^{0}$', histtype='step', hatch='//', facecolor='none', edgecolor='red') 

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
ax[1,0].scatter(np.linspace(-0.3,0.3,len(theta_x_asymmetry)), theta_x_asymmetry, color='k', marker='x')
ax[1,1].scatter(np.linspace(-0.3,0.3,len(theta_y_asymmetry)), theta_y_asymmetry, color='k', marker='x')
ax[1,2].scatter(np.linspace(0.0, 7.5e-4,len(k_asymmetry)), k_asymmetry, color='k', marker='x')

# Plot line of zeros
ax[1,0].plot(np.linspace(-0.3,0.3,len(theta_x_asymmetry)), np.zeros(len(theta_x_asymmetry)), color='r', linestyle='--')
ax[1,1].plot(np.linspace(-0.3,0.3,len(theta_y_asymmetry)), np.zeros(len(theta_y_asymmetry)), color='r', linestyle='--')
ax[1,2].plot(np.linspace(0,7.5e-4,len(k_asymmetry)), np.zeros(len(k_asymmetry)), color='r', linestyle='--')

# Adjust layout
plt.tight_layout(pad=2.0)  # Add padding between subplots to avoid overlap

for i in [0,1,2]:
    ax[0,i].set_ylabel('Counts')
    ax[0,i].legend()


plt.savefig("theta_x_y_and_k_hist_pulls.png", dpi = 500)
plt.show()

### PICKLING DATA ###
home = os.getcwd()
if (os.path.isdir('Pickles') == False):
    os.mkdir('Pickles')
os.chdir('Pickles')
file_paths = ['d0.pickle', 'db.pickle']
meson = [d_0, d_0bar]
for i in [0,1]:
    with open(file_paths[i], 'wb') as file:
        pickle.dump([meson[i]['theta_x'], meson[i]['theta_y'], meson[i]['k']], file)
os.chdir(home)
