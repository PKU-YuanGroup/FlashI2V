# FlashI2V (Ascend version)

We use [mindspeed-mm](https://gitee.com/ascend/MindSpeed-MM) and [megatron-LM](https://github.com/NVIDIA/Megatron-LM) to train FlashI2V. Please adhere to their licenses.

## ⚙️ Runtime Environment

(1) Clone FlashI2V repo.

```
git clone -b npu https://github.com/PKU-YuanGroup/FlashI2V
```

(2) Prepare the environment

```
conda create -n flashi2v python=3.10
conda activate flashi2v

# install apex for Ascend, please refer to https://gitee.com/ascend/apex

# Set the Ascend toolkit environment variables.
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# install mindspeed
git clone https://gitee.com/ascend/MindSpeed.git
cd MindSpeed
git checkout 59b4e983b7dc1f537f8c6b97a57e54f0316fafb0
pip install -r requirements.txt
pip3 install -e .
cd ..

# install other repos
pip install -e .
```

(3) install decord

```
git clone --recursive https://github.com/dmlc/decord
mkdir build && cd build 
cmake .. -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release -DFFMPEG_DIR=/usr/local/ffmpeg 
make 
cd ../python 
pwd=$PWD 
echo "PYTHONPATH=$PYTHONPATH:$pwd" >> ~/.bashrc 
source ~/.bashrc 
python3 setup.py install --user
```

## 🚙 Download Weights

You can find FlashI2V weights in https://huggingface.co/yunyangge/FlashI2V-1.3B

## 🍕 Sample Image-to-Video

```
bash examples/flashi2v/1.3b/inference.sh
```

Make sure the inference_model.json file is properly configured.
```
{
    "pipeline_class": "FlashI2VPipeline",
    "prompt": "examples/flashi2v/test_i2v/prompt.txt", # The prompt text, with each line corresponding to an image in image.txt.
    "image": "examples/flashi2v/test_i2v/image.txt", # The image.txt, which specifies the first frame image for video generation.
    "save_path": "test_samples/test_flashi2v", # Save dir
    "use_prompt_preprocess": false,
    "dtype": "bf16",
    "device": "npu",
    "frame_interval": 1,
    "fps": 16,
    "pipeline_config": {
        "input_size": [49, 480, 832]
    },
    "data_transform": {
        "video":[
            {
                "trans_type": "CenterCropResizeVideo",
                "param": {
                    "transform_size": [480, 832],
                    "align_corners": false,
                    "antialias": true
                }
            },
            {
                "trans_type": "ToTensorVideo"
            },
            {
                "trans_type": "norm_fun",
                "param": {
                    "mean": 0.5,
                    "std": 0.5
                }
            }
        ]
    },
    "ae": {
        "model_id": "wan_video_vae",
        "from_pretrained": "/work/share1/checkpoint/pretrained/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae", # Wan2.1 VAE
        "dtype": "float32",
        "do_sample": false,
        "enable_tiling": false,
        "norm_latents": true,
        "norm_mode": "channel_specified_shift_scale"
    },
    "tokenizer":{
        "autotokenizer_name": "AutoTokenizer",
        "hub_backend": "hf",
        "from_pretrained": "/work/share1/checkpoint/pretrained/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tokenizer" # Wan2.1 text tokenizer
    },
    "text_encoder": {
        "model_id": "UMT5",
        "hub_backend": "hf",
        "from_pretrained": "/work/share1/checkpoint/pretrained/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/text_encoder", # Wan2.1 text encoder
        "dtype": "bf16"
    },
    "predictor": {
        "model_id": "flashi2vdit",
        "dtype": "bf16",
        "fft_return_abs": true,
        "conv3x3x3_proj": true,
        "low_freq_proj": false,
        "low_freq_energy_ratio": 0.1, # Fourier cutoff frequency percentile, default set to 0.1.
        "patch_size": [1, 2, 2],
        "text_len": 512,
        "in_dim": 16,
        "hidden_size": 1536,
        "ffn_dim": 8960,
        "freq_dim": 256,
        "text_dim": 4096,
        "img_dim": 1280,
        "out_dim": 16,
        "num_heads": 12,
        "num_layers": 30,
        "qk_norm": true,
        "qk_norm_type": "rmsnorm",
        "cross_attn_norm": true,
        "eps": 1e-6,
        "max_seq_len": 1024,
        "use_fused_rmsnorm": true,
        "from_pretrained": "release_weights/npu/flashi2v_1_3B.pt" # pretrained weights
    },
    "diffusion": {
        "model_id": "flow_matching",
        "num_inference_steps": 50,
        "shift": 7.0,
        "guidance_scale": 5.0
    }
}
```

## 🧑‍🏭 Train Image-to-Video

### 📚 Data preparation
You should create a meta JSON for all the training videos, which includes the following information:

```
[
  {
    "path": "path/to/a/video", # Video path. This kwarg must be specified.
    "cap": "This is a caption of a video.", # Video caption. This kwarg must be specified.
    "resolution": {"height": 1080, "width": 1920}, # Video resolution. This kwarg is optional. When not explicitly specified, it retrieves the height and width of the video.
    "fps": 24, # Video fps. This kwarg is optional. When not explicitly specified, it retrieves the fps of the video.
    "num_frames": 49, # Video frame number. This kwarg is optional. When not explicitly specified, it retrieves the frame number of the entire video.
    "cut": [0, 49] # The position of the current clip within the entire video. This field is optional, designed to accommodate the case where only a segment of a long video is selected for training.
  },
  {
    "path": ...
    "cap": ...
  },
  ...
]
```

This meta JSON includes a list that records the various information about the videos used for training. 
Then, you need to specify the following code to filter your training videos to meet the requirements of different training stages.

```
python examples/flashi2v/filter_data.py --filter_config examples/flashi2v/filter_config.yaml
```

The content of `filter_config.yaml` is as follows:

```
ann_txt_path: 'examples/flashi2v/all_videos.txt' # Annotation txt of video jsons.
save_path: 'test/lmdb/all_videos_720p' # Save dir of lmdb file.
sample_height: 480 # Sample height of videos in training.
sample_width: 832 # Sample width of videos in training.
sample_num_frames: 49 # Sample frame number of videos in training.
min_hxw: 921600 # 720x1280: 921600, 480x832: 399360, 576x1024: 589824 # Min height * width of videos in training, for filtering videos with low resolution.
train_fps: 16 # Sample fps of videos in training.
max_h_div_w_ratio: 1.2 # Max H / W of videos in training.
min_h_div_w_ratio: 0.4 # Min H / W of videos in training.
```

And the content of `ann_txt_path.txt` is as follows:

```
/work,/work/share1/caption/osp/all_videos/random_video_final_1_5980186.json # Root dir of videos, meta json.
```

After filtering, we will obtain the metadata saved in LMDB format. Since LMDB allows us to maintain low memory usage when processing large datasets, it effectively avoids memory leak issues caused by the decord library.
 
### 🤾 Training

We need to first configure the files examples/flashi2v/1.3b/pretrain_model_flashi2v.json and examples/flashi2v/1.3b/data.json.

```
# pretrain_model_flashi2v.json
{
    "load_video_features": false,
    "load_text_features": false,
    "task": "t2v",
    "patch": {
        "ae_float32": "ae_float32"
    },
    "ae": {
        "model_id": "wan_video_vae",
        "from_pretrained": "/work/share1/checkpoint/pretrained/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae/", # Wan2.1 VAE
        "dtype": "fp32",
        "enable_tiling": false,
        "tiling_param": {
            "tile_size": [34, 34],
            "tile_stride": [18, 16]
        },
        "norm_latents": true,
        "norm_mode": "channel_specified_shift_scale",
        "do_sample": true
    },
    "text_encoder": {
        "model_id": "UMT5",
        "dtype": "bf16",
        "hub_backend": "hf",
        "from_pretrained": "/work/share1/checkpoint/pretrained/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/text_encoder", # Wan2.1 text encoder
        "use_attention_mask": false
    },
    "diffusion": {
        "model_id": "flow_matching",
        "use_dynamic_shifting": true,
        "use_logitnorm_time_sampling": true,
    },
    "predictor": {
        "model_id": "flashi2vdit",
        "low_freq_energy_ratio": [0.05, 0.95], # Fourier cutoff frequency percentiles, default set to [0.05,0.95] when training.
        "fft_return_abs": true, # We use Fourier magnitude as high-freq features.
        "dtype": "bf16",
        "patch_size": [1, 2, 2],
        "text_len": 512,
        "in_dim": 16,
        "hidden_size": 1536,
        "ffn_dim": 8960,
        "freq_dim": 256,
        "text_dim": 4096,
        "img_dim": 1280,
        "out_dim": 16,
        "num_heads": 12,
        "num_layers": 30,
        "pipeline_num_layers": [7,8,8,7],
        "qk_norm": true,
        "qk_norm_type": "rmsnorm",
        "cross_attn_norm": true,
        "eps": 1e-6,
        "max_seq_len": 1024,
        "attention_async_offload": false,
        "use_fused_rmsnorm": true, 
        "from_pretrained": "release_weights/npu/flashi2v_1_3B.pt" # Pretrained weights
    }
}
```

```
# data.json
{
    "dataset_param": {
        "dataset_type": "flashi2v",
        "metafile_or_dir_path": "/work/share1/caption/osp/lmdb/resi2v_2m/filtered_samples_1858822.lmdb", # lmdb path
        "text_tokenizer_path": "/work/share1/checkpoint/pretrained/models/Wan-AI/Wan2.1-T2V-1.3B-Diffusers/tokenizer/", # Wan2.1 tokenizer
        "sample_height": 480,
        "sample_width": 832,
        "sample_num_frames": 49,
        "train_fps": 16,
        "text_drop_ratio": 0.1,
        "tokenizer_max_length": 512,
        "return_prompt_mask": true,
        "transform_pipeline": {
            "video":[
                {
                    "trans_type": "CenterCropResizeVideo",
                    "param": {
                        "transform_size": [480, 832],
                        "align_corners": false,
                        "antialias": true
                    }
                },
                {
                    "trans_type": "ToTensorVideo"
                },
                {
                    "trans_type": "norm_fun",
                    "param": {
                        "mean": 0.5,
                        "std": 0.5
                    }
                }
            ]
        }
    },
    "dataloader_param":{
        "dataloader_mode": "sampler",
        "sampler_type": "StatefulDistributedSampler",
        "collate_param": {
            "model_name": "flashi2v"
        },
        "shuffle": true,
        "drop_last": true,
        "pin_memory": true,
        "num_workers": 16 # Use the maximum possible number of num_workers.
    }
}
```

Then, run the training script:

```
bash examples/flashi2v/1.3b/pretrain_ddp.sh
```
