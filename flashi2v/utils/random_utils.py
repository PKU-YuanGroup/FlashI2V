import os
import logging
import numpy as np
import random
import torch
import torch.distributed as dist
from .utils import is_npu_available

# adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/random.py#L39
def set_seed(
    seed: int,
    device_specific: bool = False, 
    deterministic: bool = False,
    process_group: dist.ProcessGroup = None
):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        device_specific (`bool`, *optional*, defaults to `False`):
            Whether to differ the seed on each device slightly with `self.process_index`.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    if device_specific:
        if process_group is None:
            if not dist.is_initialized():
                raise ValueError("`device_specific` can only be set to `True` when using distributed.")
            process_group = dist.group.WORLD
        rank = dist.get_rank(process_group)
        seed += rank
        print(f"Using device specific seed {seed} on group rank {rank} and global rank {dist.get_rank()}")

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if is_npu_available():
            import torch_npu
            torch_npu.npu.manual_seed_all(seed)
            torch_npu.npu.manual_seed(seed)
    return seed

def get_seed_worker(
    seed: int, 
    num_workers: int = 16,
    device_specific: bool = False, 
    process_group: dist.ProcessGroup = None,
):
    """Deterministic dataloader"""
    if device_specific:
        if process_group is None:
            if not dist.is_initialized():
                raise ValueError("`device_specific` can only be set to `True` when using distributed.")
            process_group = dist.group.WORLD
        rank = dist.get_rank(process_group)

    def seed_worker(worker_id):
        if device_specific:
            worker_seed = seed + rank * num_workers + worker_id # make sure all workers have different seed
        # print(f"rank = {rank}, worker_seed = {worker_seed}")
        np.random.seed(worker_seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker