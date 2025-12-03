import os

def get_dataset_paths(dataset_name: str, root_dir: str = './dataset'):
    config = {'name': dataset_name}
    if dataset_name in ['Chicago', 'NYC']:
        base_prefix = f'Taxi_{dataset_name}'
        dataset_dir = os.path.join(root_dir, base_prefix)
        config.update({
            'adj_path': os.path.join(dataset_dir, f'{base_prefix}_rn_adj.npy'),
            'train_path': os.path.join(dataset_dir, f'{base_prefix}_train1.npz'),
            'val_path': os.path.join(dataset_dir, f'{base_prefix}_val1.npz'),
            'test_path': os.path.join(dataset_dir, f'{base_prefix}_test1.npz'),
            'modal_paths': {}
        })
    elif dataset_name in ['Sacra', 'Sacra_Demo']:
        dataset_dir = os.path.join(root_dir, dataset_name)
        config.update({
            'adj_path': os.path.join(dataset_dir, 'sacramento_adj_gaussian.npy'),
            'data_path': os.path.join(dataset_dir, 'his.npz'),
            'train_idx': os.path.join(dataset_dir, 'idx_train.npy'),
            'val_idx': os.path.join(dataset_dir, 'idx_val.npy'),
            'test_idx': os.path.join(dataset_dir, 'idx_test.npy'),
            'modal_paths': {}
        })
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Please add path config in data/factory.py")
    return config