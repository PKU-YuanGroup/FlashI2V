import torch
import math

from typing import Optional
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torchdata.stateful_dataloader import Stateful
import logging

#adopted from https://github.com/meta-pytorch/data/blob/main/torchdata/stateful_dataloader/sampler.py#L18
class StatefulDistributedSampler(DistributedSampler, Stateful):

    _GENERATOR = "generator"
    _YIELDED = "yielded"

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        if self.shuffle:
            self.g = torch.Generator()
            # self.g.manual_seed(self.seed + self.epoch)
            self.g.manual_seed(self.seed)
            self.generator_state = self.g.get_state()
        self.indices = self._get_indices()
        self.next_yielded = None
        self.yielded = 0

    def _get_indices(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=self.g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples
        return indices

    def __iter__(self):
        logging.info(f"In StatefulDistributedSampler, rank = {self.rank}, first 5 indices: {self.indices[0: min(5, len(self))]}\n"
                     f"yielded samples num: {self.yielded}, current index for sampler: {self.indices[self.yielded % self.num_samples]}") # Prevent exceeding bounds when save_interval == num_stamples
        return self
    
    def __next__(self):
        if self.yielded == self.num_samples:
            self.indices = self._get_indices()
            self.yielded = 0
            logging.info(f" In StatefulDistributedSampler, data reshuffle...\n"
                         f" rank = {self.rank}, first 5 indices: {self.indices[0: min(5, len(self))]}\n"
                         f" yielded samples num: {self.yielded}, current index for sampler: {self.indices[self.yielded]}")
            raise StopIteration()
        val = self.indices[self.yielded]
        self.yielded += 1
        return val

    def state_dict(self):
        return {
            self._YIELDED: self.yielded, # yielded samples num < num samples
            self._GENERATOR: self.generator_state,
        }

    def load_state_dict(self, state_dict):
        self.next_yielded = state_dict[self._YIELDED]
        self.generator_state = state_dict[self._GENERATOR]
        self.g.set_state(self.generator_state)
        if self.next_yielded is not None:
            self.indices = self._get_indices()  # We want permutations from the latest generator state that's loaded
            self.yielded = self.next_yielded
            self.next_yielded = None



data_sampler = {
    'stateful_distributed': StatefulDistributedSampler
} 