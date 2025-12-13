import numpy as np
import os
from tqdm import tqdm

dtype = np.float64
shape = (128, 128, 64)
n_snapshots_train = 800
n_snapshots_val = 200
batch_size = 20  # merge this many snapshots at a time

variables = ['w']
data_dirs = {
    'val': 'data/ABL_Data_Evaluation'
}

for var in variables:
    for split, directory in data_dirs.items():
        n_snapshots = n_snapshots_train if split == 'train' else n_snapshots_val
        output_file = f'{directory}/{var}.npy'
        print(f"Merging {split} snapshots for variable {var} into {output_file}...")
        
        # Preallocate memmap
        all_snapshots = np.lib.format.open_memmap(output_file, mode='w+', dtype=dtype, shape=(n_snapshots, *shape))
        
        # Batch merge
        for start in tqdm(range(0, n_snapshots, batch_size), desc=f"{var} {split}"):
            end = min(start + batch_size, n_snapshots)
            batch_data = []
            for i in range(start, end):
                snapshot_file = f'{directory}/{var}_{i:04d}.npy'
                batch_data.append(np.load(snapshot_file))
            all_snapshots[start:end] = np.stack(batch_data)
            
            # Optionally delete small snapshots after merging
            #for i in range(start, end):
                #os.remove(f'{directory}/{var}_{i:04d}.npy')
        
        all_snapshots.flush()
        print(f"Finished merging {split} snapshots for {var}")
