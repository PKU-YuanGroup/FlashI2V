__all__ = [
    "build_mm_dataset", "build_mm_dataloader"
]

import copy

from torch.utils.data import ConcatDataset
from torch.distributed.distributed_c10d import _get_default_group

from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from mindspeed_mm.data.dataloader.dataloader import (
    prepare_base_dataloader,
    prepare_sampler_dataloader,
)
from mindspeed_mm.models.ae.training.global_vars import get_ae_args
from mindspeed_mm.data.datasets.wan_dataset import WanT2VDataset
from mindspeed_mm.data.datasets.flashi2v_dataset import FlashI2VDataset

def build_mm_dataset(dataset_param):
    """
    Build a multimodal dataset based on different tasks

    Args:
        dataset_param
    Return:
        dataset
    """
    if not isinstance(dataset_param, dict):
        dataset_param = dataset_param.to_dict()
    dataset_type = dataset_param.pop("dataset_type", None)
    if dataset_type == "want2v":
        return WanT2VDataset(**dataset_param)
    elif dataset_type == "flashi2v":
        return FlashI2VDataset(**dataset_param)
    else:
        raise NotImplementedError(dataset_type)


def build_mm_dataloader(dataset, dataloader_param, process_group=None, consumed_samples=0, dataset_param=None):
    """
    Build a multimodal dataloader based on different tasks

    dataloader_type interpretation:
    base: raw dataloader based on torch.utils.data.DataLoader
    sampler: prepare a dataloader for distributed training by building a specific sampler

    Args:
        dataloader_param_dict
    Return:
        dataloader
    """
    if not isinstance(dataloader_param, dict):
        dataloader_param = dataloader_param.to_dict()
    if "dataloader_mode" not in dataloader_param:
        raise AssertionError("Key parameter missing: dataloader_mode")
    dataloader_mode = dataloader_param.pop("dataloader_mode")
    if process_group is None:
        process_group = mpu.get_data_parallel_group()
    args = get_args()
    dataloader_param.update(
        {
            "batch_size": args.micro_batch_size,
            "seed": args.seed,
        }
    )
    print_rank_0(f'[INFO] initialize `batch_size`/`seed` from argument parser rather than `data.json`')
    print_rank_0(f'[INFO] `batch_size`/`seed` = {args.micro_batch_size} / {args.seed}')
    if dataloader_mode == "base":
        data_loader = prepare_base_dataloader(dataset, **dataloader_param)
        return data_loader
    elif dataloader_mode == "sampler":
        data_loader = prepare_sampler_dataloader(
            dataset, **dataloader_param, process_group=process_group, consumed_samples=consumed_samples,
            dataset_param=dataset_param
        )
        return data_loader
    else:
        raise NotImplementedError(dataloader_mode)

