import logging
import torch
import torch.nn as nn
from flashi2v.utils.utils import is_npu_available, check_and_import_npu
check_and_import_npu()
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.device_mesh import DeviceMesh

def CP_wrapper(model: nn.Module, all_cp_plans: dict, cp_mesh: DeviceMesh):
    is_rank_zero = torch.distributed.get_rank() == 0
    if is_rank_zero:
        logging.info("Parallelize Module with Context Parallel...")
    for module in model.modules():
        for module_cls, cp_plan in all_cp_plans.items():
            if isinstance(module, module_cls):
                if is_rank_zero:
                    logging.info(f"Parallelize {module_cls}.")
                parallelize_module(
                    module,
                    device_mesh=cp_mesh,
                    parallelize_plan=cp_plan
                )
    if is_rank_zero:
        logging.info("Context Parallel Down!")


if __name__ == "__main__":
    from torch.distributed.device_mesh import init_device_mesh
    from flashi2v.modules import models, models_blocks_to_float, models_main_block, models_cp_plans
    from flashi2v.distributed.utils import setup_distributed_env, cleanup_distributed_env
    from flashi2v.utils.random_utils import set_seed
    
    setup_distributed_env()

    ddp_cp_mesh = init_device_mesh(
        "cuda",
        (1, 4),
        mesh_dim_names=("ddp", "cp"),
    )
    if not is_npu_available():
        pretrained_model_dir = "/mnt/data2/Wan2.1-T2V-1.3B/"
    else:
        pretrained_model_dir = "/work/share1/checkpoint/Wan-AI/Wan2.1-T2V-1.3B/"
    print("ddp_cp_mesh:", ddp_cp_mesh)
    set_seed(1024, device_specific=True, process_group=ddp_cp_mesh["ddp"].get_group())

    model_name = "flashi2v"
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    dtype = torch.float32

    latents = torch.randn(1, 16, 16, 64, 64).to(device=device, dtype=dtype)
    start_frame_latents = torch.randn(1, 16, 16, 64, 64).to(device=device, dtype=dtype)
    fourier_features = torch.randn(1, 16, 16, 64, 64).to(device=device, dtype=dtype)
    text_embeddings = torch.randn(1, 512, 4096).to(device=device, dtype=dtype)
    timesteps = torch.randint(0, 1000, (1,)).to(device=device)

    # ddp_model = models[model_name].from_pretrained(pretrained_model_dir)
    set_seed(1024, device_specific=True, process_group=ddp_cp_mesh["ddp"].get_group())
    ddp_model = models[model_name]()
    ddp_model = torch.nn.parallel.DistributedDataParallel(ddp_model.to(device=device, dtype=dtype))
    set_seed(1024, device_specific=True, process_group=ddp_cp_mesh["ddp"].get_group())
    # cp_model = models[model_name].from_pretrained(pretrained_model_dir).to(device=device, dtype=dtype)
    cp_model = models[model_name]().to(device=device, dtype=dtype)

    CP_wrapper(cp_model, models_cp_plans[model_name], cp_mesh=ddp_cp_mesh["cp"])

    with torch.no_grad():
        ddp_output = ddp_model(latents, timesteps, text_embeddings, start_frame_latents=start_frame_latents, fourier_features=fourier_features)
        cp_output = cp_model(latents, timesteps, text_embeddings, start_frame_latents=start_frame_latents, fourier_features=fourier_features)
    # print(f"rank = {torch.distributed.get_rank()}, ddp_output[0, :10, 0]: {ddp_output[0, :10, 0]}, cp_output[0, :10, 0]: {ddp_output[0, :10, 0]}")
    print("ddp_output - cp_output MSE:", torch.mean((ddp_output.float() - cp_output.float()) ** 2))

    cleanup_distributed_env()