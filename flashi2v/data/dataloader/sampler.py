from typing import Iterator, Optional
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
import logging

class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        consumed_samples: int = 0,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0
        self.consumed_samples = consumed_samples // num_replicas
        self.epoch = (
            self.consumed_samples // self.num_samples
            if self.num_samples > 0 else 0
        )
        self.start_index = self.consumed_samples % self.num_samples
        logging.info(f'In StatefulDistributedSampler, num_samples: {self.num_samples} * num_replicas: {self.num_replicas}, '   
                f'epoch: {self.epoch}, consumed_samples: {self.consumed_samples}, start_index: {self.start_index}')


    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index:]
        logging.info(f'In StatefulDistributedSampler, first index for sampler: {indices[0]}')
        actual_indices_len = len(indices)
        self.consumed_samples += actual_indices_len
        self.epoch += 1
        self.start_index = 0
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index


data_sampler = {
    'stateful_distributed': StatefulDistributedSampler
} 