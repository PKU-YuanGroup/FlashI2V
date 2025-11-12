<p align="center">
    <img src="https://github.com/user-attachments/assets/a4284be4-c444-4d13-b8f9-d66e52655106" width="200"/>
<p>
<h2 align="center"> 
  <a href="https://arxiv.org/abs/2509.25187">
    FlashI2V: Fourier-Guided Latent Shifting Prevents Conditional Image Leakage in Image-to-Video Generation
  </a>
</h2>

<h5 align="center"> If you like our project, please give us a star ⭐ on GitHub for the latest update.  </h2>

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-FlashI2V-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2509.25187)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](https://huggingface.co/yunyangge/FlashI2V-1.3B)
[![Page](https://img.shields.io/badge/Page-GitHub-lightgrey?logo=github)](https://pku-yuangroup.github.io/FlashI2V/)

</h5>

<details open><summary>💡 We also have other generation projects that may interest you ✨. </summary><p>
<!--  may -->

> [**Open-Sora Plan: Open-Source Large Video Generation Model**](https://arxiv.org/abs/2412.00131) <br>
> Bin Lin, Yunyang Ge and Xinhua Cheng etc. <br>
[![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/Open-Sora-Plan)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/Open-Sora-Plan.svg?style=social)](https://github.com/PKU-YuanGroup/Open-Sora-Plan) [![arXiv](https://img.shields.io/badge/Arxiv-2412.00131-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2412.00131) <br>
>
> [**UniWorld-V1: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation**](https://arxiv.org/abs/2506.03147) <br>
> Bin Lin, Zongjian Li, Xinhua Cheng etc. <br>
> [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/UniWorld-V1)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/UniWorld-V1.svg?style=social)](https://github.com/PKU-YuanGroup/UniWorld-V1) [![arXiv](https://img.shields.io/badge/Arxiv-2506.03147-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.03147) <br>
>
> [**Identity-Preserving Text-to-Video Generation by Frequency Decomposition**](https://arxiv.org/abs/2411.17440) <br>
> Shenghai Yuan, Jinfa Huang, Xianyi He etc. <br>
> [![github](https://img.shields.io/badge/-Github-black?logo=github)](https://github.com/PKU-YuanGroup/ConsisID)  [![github](https://img.shields.io/github/stars/PKU-YuanGroup/ConsisID.svg?style=social)](https://github.com/PKU-YuanGroup/ConsisID) [![arXiv](https://img.shields.io/badge/Arxiv-2411.17440-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2411.17440) <br>
>
> </p></details>
   
# 📣 News
* **[2025.11.12]**  We have uploaded the FSDP2 + DeepSpeed-Ulysses CP code version, which supports both GPU (Nvidia) and NPU (Ascend) for training and inference, and is compatible with models up to 14B parameters.
* **[2025.09.30]**  We have uploaded the Ascend version of the training and inference code, along with the model weights. For details, please refer to the [NPU](https://github.com/PKU-YuanGroup/FlashI2V/tree/npu) branch.

# 🗓️ TODO
- [x] Release [paper](https://arxiv.org/abs/2509.25187)
- [x] Release [NPU(Ascend) version code](https://github.com/PKU-YuanGroup/FlashI2V/tree/npu) with [mindspeed-mm](https://gitee.com/ascend/MindSpeed-MM)
- [x] Release [page](https://pku-yuangroup.github.io/FlashI2V/)
- [x] Release [1.3B model](https://huggingface.co/yunyangge/FlashI2V-1.3B)
- [x] Release FSDP2 + DeepSpeed-Ulysses CP code version, which supports both GPU (Nvidia) and NPU (Ascend).
- [ ] Scaling FlashI2V to 14B

# 💡Usage
## ⚙️ Runtime Environment
### GPU (Nvidia)
(1) Clone FlashI2V repo.

```
git clone https://github.com/PKU-YuanGroup/FlashI2V
```

(2) Prepare the environment

```
conda create -n flashi2v python=3.10
conda activate flashi2v
```
(3) Install dependencies
```
pip install -r requirements.txt
```
(4) Install flash attn
```
pip install flash-attn --no-build-isolation
```
(5) build
```
pip install -e .
```
### NPU (Ascend)
⚠️ For proper execution of our code, please install CANN version 8.3.rc1 or later, and follow the [tutorial linked](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/softwareinst/instg/instg_0008.html?Mode=PmIns&OS=Debian&Software=cannToolKit) for detailed installation steps.
(1) Clone FlashI2V repo.

```
git clone https://github.com/PKU-YuanGroup/FlashI2V
```

(2) Prepare the environment

```
conda create -n flashi2v python=3.10
conda activate flashi2v
```
(3) Install dependencies
```
pip install -r requirements.txt
```
(4) Install decord
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
(5) build
```
pip install -e .
```
## 🍕 Sample Image-to-Video
```
bash scripts/infer/*pu/infer_flashi2v_*b.sh
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
python filter_data.py --filter_config filter_config.yaml
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
```
bash scripts/train/*pu/train_flashi2v_*b.sh
```
# 😍 Gallery
## Image-to-Video Results of FlashI2V-1.3B
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/fbf883b0-5e08-44b7-9a31-5bffc6a80125" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/9faea0ab-c726-44f1-b262-a4daa6d8a512" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/1e68967c-61ed-457d-878f-e5310a26722b" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/0abeb7b9-401b-4715-b934-986a435d8ba0" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/77f68d35-cbc2-4b3b-b371-6f17f1cfd861" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/3270ba31-56f3-4cbd-b92e-27e286a5ffb4" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/2ee57082-1150-4e97-a45b-be4885cc317b" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/8d77cfb0-d546-43d0-a717-a0dd7f3237cf" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/518af498-5488-4d3f-8401-13437d741080" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/5edc7b80-0c04-41b3-bd5c-029191bf9577" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/eb74c91e-8b4e-47b0-b4ef-6eaa149fa9ea" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/362e73a8-20b3-4f1b-a549-1e420a4ab798" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>
<table border="0" style="width: 100%; text-align: left; margin-top: 20px;">
  <tr>
      <td>
          <video src="https://github.com/user-attachments/assets/0f89c8b5-c62c-440f-ac9d-78a983a00a3e" width="100%" controls autoplay loop></video>
      </td>
      <td>
          <video src="https://github.com/user-attachments/assets/b2a020b6-f3bf-4a0f-83f7-d0194366b358" width="100%" controls autoplay loop></video>
      </td>
       <td>
          <video src="https://github.com/user-attachments/assets/418a8fa3-1f7b-4a5a-868c-70dd3dc153a8" width="100%" controls autoplay loop></video>
     </td>
      <td>
          <video src="https://github.com/user-attachments/assets/60848deb-6d22-4fba-ac9e-b168ddc2e875" width="100%" controls autoplay loop></video>
     </td>
  </tr>
</table>

# 😮 Highlights

## Overfitting to In-domain Data Causes Performance Degradation

<p align="center">
    <img src="https://github.com/user-attachments/assets/a159d190-e044-4b63-b1a3-115ebc10a7dc" style="margin-bottom: 0.2;"/>
<p>

- Existing I2V Methods involve Conditional image leakage. (a) Conditional image leakage causes performance degradation issues, where the videos are sampled from Wan2.1-I2V-14B-480P with Vbench-I2V text-image pairs. (b) In the existing I2V paradigm, we observe that chunk-wise FVD on in-domain data increases over time, while chunk-wise FVD on out-of-domain data remains consistently high, indicating that the law learned on in-domain data by the existing paradigm fails to generalize to out-of-domain data.

## Model Overview

<p align="center">
    <img src="https://github.com/user-attachments/assets/4161a4d6-021e-4eed-9667-4890c60019cf" style="margin-bottom: 0.2;"/>
<p>

- We propose FlashI2V to introduce conditions implicitly. We extract features from the conditional image latents using a learnable projection, followed by the latent shifting to obtain a renewed intermediate state that implicitly contains the condition. 
Simultaneously, the conditional image latents undergo the Fourier Transform to extract high-frequency magnitude features as guidance, which are concatenated with noisy latents and injected into DiT. During inference, we begin with the shifted noise and progressively denoise following the ODE, ultimately decoding the video.

## Best Generalization and Performance across Different I2V Paradigms

<p align="center">
    <img src="https://github.com/user-attachments/assets/07a08665-8b06-41f4-bbb7-e41d82c9371c" style="margin-bottom: 0.2;"/>
<p>
  
- Comparing the chunk-wise FVD variation patterns of different I2V paradigms on both the training and validation sets, it is observed that only FlashI2V exhibits the same time-increasing FVD variation pattern in both sets.
This suggests that only FlashI2V is capable of applying the generation law learned from in-domain data to out-of-domain data. Additionally, FlashI2V has the lowest out-of-domain FVD, demonstrating its performance advantage.

## Vbench Results

| Model                                | I2V Paradigm                        | Subject Consistency↑ | Background Consistency↑ | Motion Smoothness↑ | Dynamic Degree↑ | Aesthetic Quality↑ | Imaging Quality↑ | I2V Subject Consistency↑ | I2V Background Consistency↑ |
|--------------------------------------|-------------------------------------|----------------------|-------------------------|--------------------|-----------------|---------------------|-------------------|---------------------------|----------------------------|
| SVD-XT-1.0 (1.5B)                    | Repeating Concat and Adding Noise   | 95.52                | 96.61                   | 98.09              | 52.36           | 60.15               | 69.80             | 97.52                     | 97.63                      |
| SVD-XT-1.1 (1.5B)                    | Repeating Concat and Adding Noise   | 95.42                | 96.77                   | 98.12              | 43.17           | 60.23               | 70.23             | 97.51                     | 97.62                      |
| SEINE-512x512 (1.8B)                 | Inpainting                          | 95.28                | 97.12                   | 97.12              | 27.07           | 64.55               | **71.39**         | 97.15                     | 96.94                      |
| CogVideoX-5B-I2V                     | Zero-padding Concat and Adding Noise| 94.34                | 96.42                   | 98.40              | 33.17           | 61.87               | 70.01             | 97.19                     | 96.74                      |
| Wan2.1-I2V-14B-720P                  | Inpainting                          | 94.86                | 97.07                   | 97.90              | 51.38           | **64.75**           | 70.44             | 96.95                     | 96.44                      |
| CogVideoX1.5-5B-I2V†                 | Zero-padding Concat and Adding Noise| 95.04                | 96.52                   | **98.47**          | 37.48           | **62.68**           | **70.99**         | 97.78                     | 98.73                      |
| Wan2.1-I2V-14B-480P†                 | Inpainting                          | **95.68**            | **97.44**               | 98.46              | 45.20           | 61.44               | 70.37             | **97.83**                 | **99.08**                  |
| **FlashI2V† (1.3B)**                 | **FlashI2V**                        | 95.13                | 96.36                   | 98.35              | **53.01**       | 62.34               | 69.41             | 97.67                     | 98.72                      |

† means testing with recaptioned text-image-pairs in Vbench-I2V.


# 🔒 License
* See [LICENSE](LICENSE) for details. For Ascend version, you can see [LICENSE](https://github.com/PKU-YuanGroup/FlashI2V/blob/npu/LICENSE) in [NPU](https://github.com/PKU-YuanGroup/FlashI2V/tree/npu) branch.

# 🤝 Contributors

<a href="https://github.com/PKU-YuanGroup/FlashI2V/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PKU-YuanGroup/FlashI2V" />
</a>


# 🙏 Acknowledgements
- Wan2.1 - https://github.com/Wan-Video/Wan2.1
- Open-Sora Plan - https://github.com/PKU-YuanGroup/Open-Sora-Plan
- Mindspeed-MM - https://gitee.com/ascend/MindSpeed-MM
- Megatron-LM - https://github.com/NVIDIA/Megatron-LM

# ✏️ Citation
If you want to cite our work, please follow:
```
@misc{ge2025flashi2v,
      title={FlashI2V: Fourier-Guided Latent Shifting Prevents Conditional Image Leakage in Image-to-Video Generation}, 
      author={Yunyang Ge and Xinhua Cheng and Chengshu Zhao and Xianyi He and Shenghai Yuan and Bin Lin and Bin Zhu and Li Yuan},
      year={2025},
      eprint={2509.25187},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2509.25187}, 
}
```
