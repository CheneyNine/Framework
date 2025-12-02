import os

def get_dataset_paths(dataset_name: str, root_dir: str = './dataset'):
    config = {'name': dataset_name}
    if dataset_name in ['Chicago', 'NYC']:
        base_prefix = f'Taxi_{dataset_name}'
        config.update({
            'adj_path': os.path.join(root_dir, f'{base_prefix}_rn_adj.npy'),
            'train_path': os.path.join(root_dir, f'{base_prefix}_train1.npz'),
            'val_path': os.path.join(root_dir, f'{base_prefix}_val1.npz'),
            'test_path': os.path.join(root_dir, f'{base_prefix}_test1.npz'),
            'modal_paths': {}
        })
    else:
        raise ValueError(f"Unknown dataset name: {dataset_name}. Please add path config in data/factory.py")
    return config