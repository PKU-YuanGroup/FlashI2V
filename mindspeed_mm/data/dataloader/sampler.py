from typing import Iterator, List, Optional
import math
import logging
import random
import time
from collections import Counter, OrderedDict, defaultdict
from pprint import pformat

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.distributed import DistributedSampler
from megatron.legacy.data.data_samplers import RandomSeedDataset
from pandarallel import pandarallel
from transformers import AutoProcessor

from mindspeed_mm.data.data_utils.aspect_ratio import get_num_pixels, get_resolution_with_aspect_ratio
from mindspeed_mm.data.data_utils.utils import format_numel_str


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
        print(f'In StatefulDistributedSampler, num_samples: {self.num_samples} * num_replicas: {self.num_replicas}, '   
                f'epoch: {self.epoch}, consumed_samples: {self.consumed_samples}, start_index: {self.start_index}')


    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index:]
        print(f'In StatefulDistributedSampler, first index for sampler: {indices[0]}')
        actual_indices_len = len(indices)
        self.consumed_samples += actual_indices_len
        self.epoch += 1
        self.start_index = 0
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index



class BaseRandomBatchSampler(DistributedSampler):
    """
    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. Default: ``True``. (It is not implemented that the drop_last is false.)
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        consumed_samples: int = 0,
        data_sharding: bool = False,
    ):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.total_samples = len(dataset)
        self.micro_batch_size = batch_size
        self.consumed_samples = consumed_samples
        self.data_sharding = data_sharding
        self.epoch = 0
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * self.num_replicas
        self.last_batch_size = \
            self.total_samples % self.micro_batch_times_data_parallel_size
        if not drop_last:
            raise ValueError("It is not implemented that the drop_last is false.")

    def __len__(self):
        return self.total_samples

    def __iter__(self):
        active_total_samples = self.total_samples - self.last_batch_size
        self.epoch = self.consumed_samples // active_total_samples
        current_epoch_samples = self.consumed_samples % active_total_samples

        if isinstance(self.dataset, RandomSeedDataset):
            self.dataset.set_epoch(self.epoch)

        # data sharding and random sampling
        if self.data_sharding:
            bucket_size = (self.total_samples // self.micro_batch_times_data_parallel_size) \
                           * self.micro_batch_size
            bucket_offset = current_epoch_samples // self.num_replicas
            start_idx = self.rank * bucket_size
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.epoch)
                idx_range_bucket = torch.randperm(bucket_size, generator=g).tolist()
            else:
                idx_range_bucket = list(range(bucket_size))
            idx_range = [start_idx + x for x in idx_range_bucket[bucket_offset:]]
        else:
            full_bucket_size = (self.total_samples // self.micro_batch_size) \
                                * self.micro_batch_size
            full_bucket_offset = current_epoch_samples
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.epoch)
                idx_range_total = \
                    torch.randperm(full_bucket_size, generator=g).tolist()
            else:
                idx_range_total = list(range(full_bucket_size))
            idx_range_active = idx_range_total[full_bucket_offset:]
            idx_range = idx_range_active[self.rank::self.num_replicas]

        batch = []
        # Last batch if not complete will be dropped.
        for idx in idx_range:
            batch.append(idx)
            if len(batch) == self.micro_batch_size:
                self.consumed_samples += self.micro_batch_times_data_parallel_size
                yield batch
                batch = []
