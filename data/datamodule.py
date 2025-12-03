import lightning as L
import torch
import numpy as np
import os
import pickle
import torch.distributed as dist
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any

from .dataset import MultimodalDataset
from .sampler import MultiSourceBatchSampler

class MultiDataModule(L.LightningDataModule):
    def __init__(self,
                 dataset_configs: List[Dict[str, Any]],
                 batch_size: int = 32,
                 history_seq_len: int = 12,
                 future_seq_len: int = 12,
                 stride: int = 1,
                 num_workers: int = 4,
                 use_modalities: bool = False):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_configs = dataset_configs
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.scalers = []

        self.in_channels = 1
        self.num_timesteps_input = history_seq_len
        self.num_timesteps_output = future_seq_len

        if len(dataset_configs) > 0:
            root_dir = os.path.dirname(dataset_configs[0]['adj_path'])
            self.scaler_dir = os.path.join(root_dir, 'scalers')
            os.makedirs(self.scaler_dir, exist_ok=True)
        else:
            self.scaler_dir = './scalers'

    def setup(self, stage=None):
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.scalers = []

        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

        for config in self.dataset_configs:
            name = config.get('name', 'Unknown')
            adj_path = config['adj_path']

            if not os.path.exists(adj_path):
                raise FileNotFoundError(f"Adjacency matrix not found for {name}: {adj_path}")
            adj = np.load(adj_path)

            dataset_common_args = {
                "adj": adj,
                "history_seq_len": self.hparams.history_seq_len,
                "future_seq_len": self.hparams.future_seq_len,
                "stride": self.hparams.stride,
                "modal_data_paths": config.get('modal_paths', {}),
                "use_modalities": self.hparams.use_modalities
            }

            scaler_path = os.path.join(self.scaler_dir, f'{name}_scaler.pkl')
            scaler = None

            if stage == 'fit' or stage is None:
                train_raw = np.load(config['train_path'])['data'].astype(np.float32)
                print(f"Dataset {name} - Train data shape: {train_raw.shape}")
                scaler = StandardScaler()
                train_norm = scaler.fit_transform(train_raw)
                self.scalers.append(scaler)

                if rank == 0:
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(scaler, f)

                self.train_datasets.append(
                    MultimodalDataset(data=train_norm, **dataset_common_args)
                )

                val_raw = np.load(config['val_path'])['data'].astype(np.float32)
                print(f"Dataset {name} - Val data shape: {val_raw.shape}")
                val_norm = scaler.transform(val_raw)
                self.val_datasets.append(
                    MultimodalDataset(data=val_norm, **dataset_common_args)
                )

            if stage == 'test' or stage is None:
                if scaler is None:
                    if os.path.exists(scaler_path):
                        with open(scaler_path, 'rb') as f:
                            scaler = pickle.load(f)
                if scaler is None:
                    train_raw_for_fit = np.load(config['train_path'])['data'].astype(np.float32)
                    scaler = StandardScaler()
                    scaler.fit(train_raw_for_fit)

                if not any(s == scaler for s in self.scalers):
                    self.scalers.append(scaler)

                test_raw = np.load(config['test_path'])['data'].astype(np.float32)
                print(f"Dataset {name} - Test data shape: {test_raw.shape}")
                test_norm = scaler.transform(test_raw)
                self.test_datasets.append(
                    MultimodalDataset(data=test_norm, **dataset_common_args)
                )

    def collate_fn(self, batch):
        xs, ys, adjs = zip(*batch)
        xs = torch.stack(xs, 0)
        ys = torch.stack(ys, 0)
        adj = adjs[0]
        return xs, ys, adj

    def train_dataloader(self):
        if not self.train_datasets:
            return None

        concat_dataset = ConcatDataset(self.train_datasets)
        dataset_sizes = [len(ds) for ds in self.train_datasets]

        num_replicas, rank = (dist.get_world_size(),
                              dist.get_rank()) if dist.is_available() and dist.is_initialized() else (1, 0)

        batch_sampler = MultiSourceBatchSampler(
            dataset_sizes=dataset_sizes,
            batch_size=self.hparams.batch_size,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=True,
            drop_last=True
        )

        return DataLoader(
            concat_dataset,
            batch_sampler=batch_sampler,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        if not self.val_datasets: return None
        return [DataLoader(ds,
                           batch_size=self.hparams.batch_size,
                           shuffle=False,
                           num_workers=self.hparams.num_workers,
                           collate_fn=self.collate_fn)
                for ds in self.val_datasets]

    def test_dataloader(self):
        if not self.test_datasets: return None
        return [DataLoader(ds,
                           batch_size=self.hparams.batch_size,
                           shuffle=False,
                           num_workers=self.hparams.num_workers,
                           collate_fn=self.collate_fn)
                for ds in self.test_datasets]