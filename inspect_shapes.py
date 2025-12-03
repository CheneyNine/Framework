import numpy as np
import os

def print_shape(filepath):
    if not os.path.exists(filepath):
        print(f"{filepath}: File not found")
        return

    try:
        if filepath.endswith('.npy'):
            data = np.load(filepath)
            print(f"{filepath}: {data.shape}")
        elif filepath.endswith('.npz'):
            data = np.load(filepath)
            print(f"{filepath}:")
            for key in data.files:
                print(f"  - {key}: {data[key].shape}")
    except Exception as e:
        print(f"{filepath}: Error loading - {e}")

files_to_check = [
    'dataset/sacramento_data.npy',
    'dataset/Sacra/his.npz',
    'dataset/Sacra/idx_train.npy',
    'dataset/Sacra/idx_val.npy',
    'dataset/Sacra/idx_test.npy',
    'dataset/Sacra/sacramento_adj_gaussian.npy',
    'dataset/Taxi_Chicago/Taxi_Chicago_train1.npz',
    'dataset/Taxi_Chicago/Taxi_Chicago_val1.npz',
    'dataset/Taxi_Chicago/Taxi_Chicago_test1.npz',
    'dataset/Taxi_Chicago/Taxi_Chicago_rn_adj.npy',
    'dataset/Taxi_NYC/Taxi_NYC_train1.npz',
    'dataset/Taxi_NYC/Taxi_NYC_val1.npz',
    'dataset/Taxi_NYC/Taxi_NYC_test1.npz',
    'dataset/Taxi_NYC/Taxi_NYC_rn_adj.npy'
]

for f in files_to_check:
    print_shape(f)
