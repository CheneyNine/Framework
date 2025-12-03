import os
import argparse
import numpy as np
import pandas as pd

def generate_data(args):
    # 1. Load Data
    data_path = '/root/code/Framework/dataset/sacramento_data.npy'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    data = np.load(data_path)
    print(f"Original data shape: {data.shape}") # (105120, 517, 4)
    
    num_samples, num_nodes, feature_dim = data.shape
    
    # 2. Create Time Index (Assuming 2023-01-01 start, 5min freq)
    # Adjust this if the actual start time is different
    full_start_time = '2023-01-01 00:00:00'
    freq = '5T'
    datetime_index = pd.date_range(start=full_start_time, periods=num_samples, freq=freq)
    
    # 3. Filter by Date
    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1) # Include the end date fully
    
    mask = (datetime_index >= start_date) & (datetime_index <= end_date)
    data = data[mask]
    datetime_index = datetime_index[mask]
    
    print(f"Filtered data shape ({args.start_date} to {args.end_date}): {data.shape}")
    
    if len(data) == 0:
        raise ValueError("No data selected. Check your start_date and end_date.")

    # 4. Add Features (Time of Day, Day of Week)
    feature_list = [data]
    
    if args.tod:
        # Time of day: normalized to [0, 1]
        time_ind = (datetime_index.values - datetime_index.values.astype('datetime64[D]')) / np.timedelta64(1, 'D')
        time_of_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_of_day)
        
    if args.dow:
        # Day of week: normalized to [0, 1]
        dow = datetime_index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        day_of_week = dow_tiled / 7
        feature_list.append(day_of_week)

    data = np.concatenate(feature_list, axis=-1)
    print(f"Data shape after feature engineering: {data.shape}")

    # 5. Split Data (Train/Val/Test)
    num_samples = len(data)
    num_train = round(num_samples * 0.6)
    num_val = round(num_samples * 0.2)
    num_test = num_samples - num_train - num_val

    train_data = data[:num_train]
    val_data = data[num_train: num_train + num_val]
    test_data = data[num_train + num_val:]
    
    train_time = datetime_index[:num_train]
    val_time = datetime_index[num_train: num_train + num_val]
    test_time = datetime_index[num_train + num_val:]

    # 6. Save Data
    out_dir = os.path.join('/root/code/Framework/dataset', args.output_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # Node indices (0 to num_nodes-1)
    node_idx = np.arange(num_nodes)
    
    # Helper to save in Taxi_Chicago format but with extra feature dim
    def save_split(filename, data_split, time_split):
        save_path = os.path.join(out_dir, filename)
        # Convert timestamps to string format for compatibility if needed, or keep as is.
        # Taxi_Chicago uses strings like '2019-01-01 00:00:00'
        time_str = time_split.strftime('%Y-%m-%d %H:%M:%S').values.astype(str)
        
        np.savez_compressed(
            save_path,
            data=data_split,
            index=node_idx,
            start_time=time_str
        )
        print(f"Saved {save_path}: data={data_split.shape}")

    save_split(f'{args.dataset_name}_train.npz', train_data, train_time)
    save_split(f'{args.dataset_name}_val.npz', val_data, val_time)
    save_split(f'{args.dataset_name}_test.npz', test_data, test_time)
    
    # Also save the adjacency matrix if it exists in the source folder to the new folder
    # Assuming source adj is 'dataset/Sacra/sacramento_adj_gaussian.npy'
    src_adj = '/root/code/Framework/dataset/Sacra/sacramento_adj_gaussian.npy'
    if os.path.exists(src_adj):
        dst_adj = os.path.join(out_dir, f'{args.dataset_name}_adj.npy')
        import shutil
        shutil.copy(src_adj, dst_adj)
        print(f"Copied adjacency matrix to {dst_adj}")

if __name__ == '__main__':
    # How to use: python data/generate_data_for_training.py --start_date 2023-01-01 --end_date 2023-01-31 --output_dir Sacra_Jan2023 --dataset_name Sacra_Jan
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='Sacra_New', help='Output dataset name prefix')
    parser.add_argument('--output_dir', type=str, default='Sacra_New', help='Output directory name inside dataset/')
    parser.add_argument('--start_date', type=str, default='2023-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2023-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--tod', type=int, default=1, help='Add time of day feature (1/0)')
    parser.add_argument('--dow', type=int, default=1, help='Add day of week feature (1/0)')
    
    args = parser.parse_args()
    generate_data(args)
