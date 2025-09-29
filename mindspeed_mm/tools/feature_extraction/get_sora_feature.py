import json
import os
import time
import uuid

import mindspeed.megatron_adaptor
import torch
import torch.distributed
from megatron.core import mpu
from megatron.training import get_args, print_rank_0
from megatron.training.initialize import initialize_megatron, set_jit_fusion_options
from numpy import save

from mindspeed_mm.configs.config import merge_mm_args, mm_extra_args_provider
from mindspeed_mm.data import build_mm_dataloader, build_mm_dataset
from mindspeed_mm.data.data_utils.constants import (
    PROMPT_IDS,
    PROMPT_IDS_2,
    PROMPT_MASK,
    PROMPT_MASK_2,
    VIDEO,
    VIDEO_MASK,
)
from mindspeed_mm.models.ae import AEModel
from mindspeed_mm.models.text_encoder import TextEncoder
from mindspeed_mm.tools.profiler import Profiler
from mindspeed_mm.utils.utils import get_device, get_dtype, is_npu_available

if is_npu_available():
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
    torch.npu.config.allow_internal_format = False


def prepare_model(args, device):
    vae = AEModel(args.mm.model.ae).to(device, args.mm.model.ae.dtype).eval()
    text_encoder = TextEncoder(args.mm.model.text_encoder).to(device).eval()
    return vae, text_encoder


def extract_feature():
    
    initialize_megatron(extra_args_provider=mm_extra_args_provider, args_defaults={})
    args = get_args()
    merge_mm_args(args)

    extract_video_feature = args.mm.tool.sorafeature.extract_video_feature
    extract_text_feature = args.mm.tool.sorafeature.extract_text_feature
    data_storage_mode = args.mm.tool.sorafeature.data_storage_mode
    
    save_path = args.mm.tool.sorafeature.save_path
    
    if torch.distributed.get_rank() == 0:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if data_storage_mode == 'standard':
            if not os.path.exists(os.path.join(save_path, 'videos')):
                os.makedirs(os.path.join(save_path, 'videos'))
            if not os.path.exists(os.path.join(save_path, 'labels')):
                os.makedirs(os.path.join(save_path, 'labels'))
            if not os.path.exists(os.path.join(save_path, 'images')):
                os.makedirs(os.path.join(save_path, 'images'))

    
    set_jit_fusion_options()
    torch.set_grad_enabled(False)
    dtype = get_dtype(args.mm.model.ae.dtype)
    device = get_device("npu")

    train_dataset = build_mm_dataset(args.mm.data.dataset_param)
    train_dataloader = build_mm_dataloader(
        train_dataset,
        args.mm.data.dataloader_param,
        process_group=mpu.get_data_parallel_group(),
    )
    
    vae, text_encoder = prepare_model(args, device)
    
    start_time = time.time()
    print_rank_0(f"Features extraction begins. {len(train_dataloader)} data in total.")
    counter = 0
    prof = Profiler(args.mm.tool.profile)
    prof.start()
    for batch in train_dataloader:
        counter += 1
        
        video = batch.pop(VIDEO).to(device, dtype)
        prompt_ids = batch.pop(PROMPT_IDS).to(device)
        prompt_mask = batch.pop(PROMPT_MASK).to(device)
        
        
        if extract_video_feature:
            latents, latents_dict = vae.encode(video)
        else:
            latents = video
        
        if extract_text_feature:
            prompt, prompt_mask = text_encoder.encode(prompt_ids, prompt_mask)
        else:
            prompt = prompt_ids
        
        if torch.distributed.get_rank() == 0:
            if data_storage_mode == 'standard':
                loca = time.strftime("%Y-%m-%d-%H-%M-%S")
                pt_name = "feature" + loca + "-" + uuid.uuid4().hex + ".pt"
                torch.save(latents.squeeze(0), os.path.join(save_path, 'videos', pt_name))
                torch.save(prompt.squeeze(0), os.path.join(save_path, 'labels', pt_name))
                data_to_save = {
                    "file": os.path.join('videos', pt_name),
                    "captions": os.path.join('labels', pt_name)
                }
                if latents_dict is not None:
                    for k in latents_dict:
                        latents_dict[k] = latents_dict[k].squeeze(0)
                    torch.save(latents_dict, os.path.join(save_path, 'images', pt_name))
                    data_to_save["image_latent"] = os.path.join('images', pt_name)
                with open(os.path.join(save_path, 'data.jsonl'), 'a', encoding="utf-8") as json_file:
                    json_file.write(json.dumps(data_to_save) + '\n')
                print_rank_0(f"consumed sample {counter} | elapsed time {(time.time() - start_time):.2f} | file {pt_name}")
            else:
                data_to_save = {
                    "latents": latents.squeeze(0),
                    "prompt": prompt.squeeze(0),
                    "prompt_mask": prompt_mask.squeeze(0)
                }
                
                loca = time.strftime('%Y-%m-%d-%H-%M-%S')
                pt_name = "feature" + loca + "-" + uuid.uuid4().hex + ".pt"
                torch.save(data_to_save, os.path.join(save_path, pt_name))
        prof.step()
    prof.stop()

    duration = time.time() - start_time
    print_rank_0(f"{counter} feature vectors extracted in {duration:.2f} seconds.")


if __name__ == "__main__":
    extract_feature()
