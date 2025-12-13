import numpy as np
import os
from tqdm import tqdm

dtype = np.float64
shape = (128, 128, 64)
n_snap_total = 1000
n_snapshots_train = 800
directory_data = '/work/wire/guest/ABL-dataset'

variables = {'p': 'p', 'u': 'ux', 'v': 'uy', 'w': 'uz'}

os.makedirs('data/ABL_Data_Training', exist_ok=True)
os.makedirs('data/ABL_Data_Evaluation', exist_ok=True)

for var, prefix in variables.items():
    
    # Training snapshots
    train_file = f'data/ABL_Data_Training/{var}.npy'
    print(f"Saving training data for {var} incrementally...")
    for i in tqdm(range(n_snapshots_train), desc=f"Training {var}"):
        filename = f"{prefix}{i+1:04d}"
        filepath = os.path.join(directory_data, filename)
        data = np.fromfile(filepath, dtype=dtype).reshape(shape, order='F')
        # Save one snapshot at a time
        np.save(f'{train_file[:-4]}_{i:04d}.npy', data)
    
    # Validation snapshots
    val_file = f'data/ABL_Data_Evaluation/{var}.npy'
    print(f"Saving validation data for {var} incrementally...")
    for i in tqdm(range(n_snapshots_train, n_snap_total), desc=f"Validation {var}"):
        filename = f"{prefix}{i+1:04d}"
        filepath = os.path.join(directory_data, filename)
        data = np.fromfile(filepath, dtype=dtype).reshape(shape, order='F')
        np.save(f'{val_file[:-4]}_{i - n_snapshots_train:04d}.npy', data)
