import torch
import numpy as np
import math
from torch.utils.data import Sampler
from typing import List, Iterator

class MultiSourceBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset_sizes: List[int], batch_size: int,
                 num_replicas: int = 1, rank: int = 0,
                 shuffle: bool = True, drop_last: bool = False):
        self.dataset_sizes = dataset_sizes
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.boundaries = [0] + np.cumsum(dataset_sizes).tolist()

    def __iter__(self) -> Iterator[List[int]]:
        all_batches = []
        g = torch.Generator()
        g.manual_seed(self.epoch)

        for i in range(len(self.dataset_sizes)):
            start_idx = self.boundaries[i]
            size = self.dataset_sizes[i]

            indices = torch.arange(size)
            if self.shuffle:
                indices = indices[torch.randperm(size, generator=g)]

            global_indices = indices + start_idx
            global_indices = global_indices.tolist()

            for j in range(0, size, self.batch_size):
                batch = global_indices[j: j + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            np.random.seed(self.epoch)
            np.random.shuffle(all_batches)

        num_samples = len(all_batches)
        if self.drop_last and num_samples % self.num_replicas != 0:
            num_samples = (num_samples // self.num_replicas) * self.num_replicas
            all_batches = all_batches[:num_samples]

        subs_batches = all_batches[self.rank: num_samples: self.num_replicas]
        return iter(subs_batches)

    def __len__(self):
        total_batches = 0
        for size in self.dataset_sizes:
            if self.drop_last:
                total_batches += size // self.batch_size
            else:
                total_batches += (size + self.batch_size - 1) // self.batch_size

        if self.drop_last:
            return total_batches // self.num_replicas
        else:
            return math.ceil(total_batches / self.num_replicas)

    def set_epoch(self, epoch: int):
        self.epoch = epoch