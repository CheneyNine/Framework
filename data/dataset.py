import torch
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Optional, Tuple

class MultimodalDataset(Dataset):
    def __init__(self, data: np.ndarray, adj: np.ndarray,
                 history_seq_len: int, future_seq_len: int,
                 stride: int = 1,
                 modal_data_paths: Optional[Dict[str, str]] = None,
                 use_modalities: bool = False):
        self.data = data
        self.adj = torch.from_numpy(adj).float()
        self.history_len = history_seq_len
        self.future_len = future_seq_len
        self.stride = stride
        self.use_modalities = use_modalities
        self.modal_data = {} # Assuming logic for loading modal data is here if needed

        total_len = len(data) - self.history_len - self.future_len + 1
        self.num_samples = (total_len + self.stride - 1) // self.stride

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Tuple:
        start_idx = idx * self.stride
        mid_idx = start_idx + self.history_len
        end_idx = mid_idx + self.future_len

        x_seq = self.data[start_idx:mid_idx, :]
        y_seq = self.data[mid_idx:end_idx, :]

        x_seq = np.expand_dims(x_seq.T, axis=1)
        y_seq = np.expand_dims(y_seq.T, axis=1)

        x_tensor = torch.from_numpy(x_seq).float()
        y_tensor = torch.from_numpy(y_seq).float()

        if self.use_modalities:
            return (
                x_tensor,
                y_tensor,
                self.adj,
                self.modal_data.get('poi'),
                self.modal_data.get('satellite'),
                self.modal_data.get('location')
            )
        else:
            return x_tensor, y_tensor, self.adj