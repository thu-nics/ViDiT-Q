<div align="center">
<h1> <img src="https://github.com/A-suozhang/MyPicBed/raw/master/img/20240624211802.png" alt="drawing" width="30"/> ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation</h1>  

<a href="https://arxiv.org/abs/2406.02540">
  <img alt="arxiv" src="https://img.shields.io/badge/arXiv-%3C2406.02540%3E-%23a72f20.svg">
</a>
<a href="https://a-suozhang.xyz/viditq.github.io/">
    <img alt="Project Page" src="https://img.shields.io/badge/Project_Page-blue?style=flat&logo=googlechrome&logoColor=white">
</a>
</div>


### News

- [24/07] We release the ViDiT-Q algorithm-level quantization simulation code at [https://github.com/thu-nics/ViDiT-Q](https://github.com/thu-nics/ViDiT-Q).

---

This repo contains the official code of [ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation](https://arxiv.org/abs/2406.02540)

We introduce ViDiT-Q, a quantization method specialized for diffusion transformers. For popular large-scale models (e.g., open-sora, Latte, Pixart-α, Pixart-Σ) for the video and image generation task, ViDiT-Q could achieve W8A8 quantization without metric degradation, and W4A8 without notable visual quality degradation.

![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20240624213022.png)

ViDiT-Q could generate videos/images with negligible discrepancy with the FP16 baseline:

| **FP16** | **Baseline Quant W8A8**  |  **ViDiT-Q W8A8** |
|---|---|---|
| <img src="./assets/forest/sample_7_fp16.gif" width=""> | <img src="./assets/forest/sample_7_baseline_w8a8.gif" width=""> | <img src="./assets/forest/sample_7_vidit_q_w8a8.gif" width=""> |
| <img src="./assets/turtles/fp16.gif" width=""> | <img src="./assets/turtles/naive_ptq_W8A8.gif" width=""> | <img src="./assets/turtles/vidit_q_W8A8.gif" width="">  |

![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20240624215505.png)

For more information, please refer to our [Project Page: https://a-suozhang.xyz/viditq.github.io/](https://a-suozhang.xyz/viditq.github.io/)

# Env Setup

We recommend using conda for enviornment management. 

```shell 
cd diffuser-dev

# create a virtual env
conda create -n viditq python=3.10
# activate virtual environment
conda activate viditq

# the xformers (opensora requires) requires torch version of 2.1.1, newest torch is not compatible
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia  # pip install also works

pip install -r t2i/requirements_pixart.txt

pip install -r t2v/requirements_opensora.txt

pip install -r t2v/requirements_qdiff.txt

# install flash attention (optional)
pip install packaging ninja
pip install flash-attn --no-build-isolation

# install xformers
pip3 install xformers==0.0.23

# install the viditq package
# containing our qdiff
pip install -e .

# install opensora
cd t2v
pip install -e .
```

<br>

# Commands to Run

> *After running the following commands, the output (ckpt,generated videos) will appear in the `./logs/`.*

We provide the shell scripts for all process below in `t2i/shell_scripts` and `t2v/shell_scripts`.
For example, run `bash t2v/shell_scripts/get_calib_data.sh $GPU_ID` to generate the calibration dataset.

## 🎬 video generation

### 0.0 Download and convert checkpoint of the STDiT (OpenSORA) model

> Please ref [doc of open-sora v1.0](https://github.com/hpcaitech/Open-Sora) for more details, we only support OpenSORA v1.0 for now, newer versions will be further supported.

- Download the OpenSora-v1-HQ-16x512x512.pth from [this link](https://huggingface.co/hpcai-tech/Open-Sora/blob/main/OpenSora-v1-HQ-16x512x512.pth)

> the original opensora code merges the qkv linears into a linear layer with more channels, we split it into 3 layers for quantization. 

- Put the downloaded OpenSora-v1-HQ-16x512x512.pth in `./logs/split_ckpt`, and run `t2v/scripts/split_ckpt.py`, the converted checkpoint will appear in `./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split-test.pth'`. 

```shell
python t2v/scripts/split_ckpt.py
```

### 0.1. FP16 inference

- `bash ./t2v/shell_scripts/fp16_inference.sh $GPU_ID`: conducting FP16 inference to generate videos using the 10 opensora example prompt, the video will be saved at `./logs/fp16_inference`. 

> we provide the precomputed `text_embeds.pth` for 10 opensora example prompts in `t2v/util_files`, which help to avoid loading the t5 ckpts onto GPU (which takes around 1 min, and around 10 GBs of memory) . Please add `--precompute_text_embeds ./t2v/utils_files/text_embeds.pth` when running command.

```shell
CFG="./t2v/configs/opensora/inference/16x512x512.py"  # the opensora config
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"  # your path of splited ckpt
OUTDIR="./logs/fp16_inference"  # your_path_to_save_videos
GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID python t2v/scripts/inference.py $CFG --ckpt_path $CKPT_PATH  --outdir $OUTDIR \
--precompute_text_embeds ./t2v/utils_files/text_embeds.pth
```

---

### 1.1 Generate calib data

- `bash ./t2v/shell_scripts/get_calib_data.sh $GPU_ID`: generating the calibration data (store the activations) at `$CALIB_DATA_DIR/calib_data.pt` for PTQ. 

```shell
CFG="./t2v/configs/opensora/inference/16x512x512.py" # the opensora config
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"  # splited ckpt
GPU_ID=$1
CALIB_DATA_DIR="./logs/calib_data"  # the path to save your calib dataset

# quant calib data
CUDA_VISIBLE_DEVICES=$GPU_ID python t2v/scripts/get_calib_data.py $CFG --ckpt_path $CKPT_PATH --data_num 10 --outdir $CALIB_DATA_DIR --save_dir $CALIB_DATA_DIR \
--precompute_text_embeds ./t2v/utils_files/text_embeds.pth
```

### 1.2 Post Training Quantization (PTQ) Process

> We provide configs for different quantizaiton techniques, for details of these configs, please refer to [our paper](https://arxiv.org/abs/2406.02540). It's worth noting that some techniques are compatible but not applied for higher bitwidth (e.g., W8A8) for simplicity. 

- `bash ./t2v/shell_scripts/ptq.sh $GPU_ID`: conducting the PTQ process based on calib data, generate the quantized checkpoint, remember to modify the names for configs and output log:
    - `CFG`: the configuration for opensora inference (we recommend using the same for calib_data generation, PTQ, and quant infernece)
    - `Q_CFG`: the configurations for quantization, we provide example configs in `./t2v/configs/quant/opensora`
        - `w8a8_naive.yaml`: Naive PTQ, tensor-wise static activation quant params
        - `w8a8_dynamic.yaml`: Dynamic Quant, token-wise, dynamic activation quant params
        - `w6a6_naive_cb.yaml`: Dynamic Quant + Naive "smooth quant"-like channel balancing (W8A8 performs relatively good without channel balancing, we use W6A6 to demonstrate the effectiveness of channel balancing)
        - `w4a8_naive_cb.yaml`: Dynamic Quant + Naive "smooth quant"-like channel balancing (naive "smoothquant"-like channel balancing works on W6A6, but fails on W4A8)
        - `w4a8_timestep_aware_cb.yaml`: Dynamic Quant + Timestep-aware channel balancing 
    - `CALIB_DATA_DIR`: the path of calibration data
    - `OUTDIR`: the path of outputs, including quantized checkpoint and copied configs

- We show the correspondence between the "ViDiT-Q" plans in the paper and config files as follows:

| Plan | CFG Name | 
|---|---|
| **ViDiT-Q W8A8**  | `w8a8_dynamic.yaml`  |
| **ViDiT-Q W6A6**  | `w6a6_naive_cb.yaml`  |
| **ViDiT-Q W4A8**  | `w4a8_timestep_aware_cb.yaml`  |

- the `--part_fp` denotes skip the quantization of a few layers (they only account for a negligible amount of computation (<1%)), the arg is defined in `opensora/utils/config_utils.py`, which reads the `part_fp_list` in quant config (default path is `"./t2v/configs/quant/opensora/remain_fp.txt"`). 

```  shell
EXP_NAME="w8a8_naive"

CFG="./t2v/configs/quant/opensora/16x512x512.py"  # the opensora config
Q_CFG="./t2v/configs/quant/opensora/$EXP_NAME.yaml"  # TODO: the config of PTQ
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"  # splited ckpt generated by split_ckpt.py
CALIB_DATA_DIR="./logs/calib_data"  # your path of calib data
OUTDIR="./logs/$EXP_NAME"  # TODO: your path to save the ptq result
GPU_ID=$1

# ptq
CUDA_VISIBLE_DEVICES=$GPU_ID python t2v/scripts/ptq.py $CFG --ckpt_path $CKPT_PATH --ptq_config $Q_CFG --outdir $OUTDIR \
    --calib_data $CALIB_DATA_DIR/calib_data.pt \
    --part_fp \
    --precompute_text_embeds ./t2v/utils_files/text_embeds.pth

```

### 1.3 Quantized Model Inference

#### 1.3.1 normal quantized inference

- `bash ./t2v/shell_scripts/quant_inference.sh $GPU_ID`: conduct the quantized model inference based on the existing quant config and quantized checkpoint (specified by the `OUTDIR`, which is the output path of the PTQ process). 

```shell
EXP_NAME="w8a8_naive"

CFG="./t2v/configs/quant/opensora/16x512x512.py" # the opensora config
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"  # your path of splited ckpt
OUTDIR="./logs/$EXP_NAME"  # your path of the w8a8 ptq result
GPU_ID=$1
# SAVE_DIR="W8A8_ptq"  # your path to save generated, leave blank to save at $OUTDIR/generated_videos

# quant inference
CUDA_VISIBLE_DEVICES=$GPU_ID python t2v/scripts/quant_txt2video.py $CFG \
    --outdir $OUTDIR --ckpt_path $CKPT_PATH  \
    --dataset_type opensora \
    --part_fp \
    --precompute_text_embeds ./t2v/utils_files/text_embeds.pth \
    # --save_dir $SAVE_DIR \

```

#### 1.3.2 mixed precision quantized inference

- `bash ./t2v/shell_scripts/quant_inference_mp.sh $GPU_ID`: conduct mixed precision quantized model inference based on the existing quant config and quantized checkpoint (specified by the `OUTDIR`, which is the output path of the PTQ process), and the mixed precision configurations `MP_W_CONFIG`, `MP_A_CONFIG` (the bit-width configuration is determined with heuristic decision based on metric-decoupled sensitivity). The code presents the  🔑 **ViDiT-Q W4A8-MP** in our paper. 

- During the PTQ process, quantization parameters for all bitwidth (4,6,8) within the quant config are calculated. Therefore, one could pair the same quantized checkpoint with differnt mixed precision configurations. 

```shell
EXP_NAME='w4a8_timestep_cb'

CFG="./t2v/configs/quant/opensora/16x512x512.py" # the opensora config
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"  # splited ckpt generated by split_ckpt.py
OUTDIR="./logs/$EXP_NAME"  # the path of the result of the W4A8 PTQ
GPU_ID=$1
MP_W_CONFIG="./t2v/configs/quant/W4A8_Naive_Smooth/t20_weight_4_mp.yaml"  # the mixed precision config of weight
MP_A_CONFIG="./t2v/configs/quant/W4A8_Naive_Smooth/t20_act_8_mp.yaml" # the mixed precision config of act
#SAVE_DIR="W4A8_Naive_Smooth_samples"  # leave blank to use the default path $OUTDIR/generated_videos

# quant infer
CUDA_VISIBLE_DEVICES=$GPU_ID python t2v/scripts/quant_txt2video_mp.py $CFG --outdir $OUTDIR --ckpt_path $CKPT_PATH  --dataset_type opensora \
	--part_fp\
	--timestep_wise_mp \
	--time_mp_config_weight $MP_W_CONFIG \
	--time_mp_config_act $MP_A_CONFIG \
	--precompute_text_embeds ./t2v/utils_files/text_embeds.pth \
	#--save_dir $SAVE_DIR
```

### 1.3.3. Get Sensitivity (optional)

- `bash ./t2v/shell_scripts/get_sensitivity.sh $GPU_ID`: get the sensitivity of certain layer. We measure the sensitivity of certain layer (block) by solely quantizing them, and measure its influence on output features. 

 ```shell
# get the sensitivity through the sensitivity
CFG="./t2v/configs/quant/opensora/16x512x512.py" # the opensora config
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"  # splited ckpt
TEXT_EMBEDS_OUTDIR="./logs/text_embeds/opensora"
OUTDIR="./logs/w8a8_ptq"  # your path of the w8a8 ptq result
GPU_ID=$1
PRE_COMPUTED_TEXTEMB="./t2v/utils_files/text_embeds.pth"
quant_group='.attn.'  # we split the model in to 4 groups: ['.attn.', 'attn_temp', 'cross_attn', 'mlp']
SAVE_PATH="w8a8_sensitivity_$quant_group"  # your path to save generated videos

# timestep wise quant + block wise quant + group wise quant
python t2v/scripts/get_sensitivity.py $CFG --ckpt_path $CKPT_PATH --outdir $OUTDIR --save_dir $SAVE_PATH --dataset_type opensora --precompute_text_embeds $PRE_COMPUTED_TEXTEMB --part_fp \
--block_group_wise_quant --timestep_wise_quant --group_quant quant_group
```

<br>

## 🖼️ image generation

### 0.0 Downloading model weights
Download the corresponding model weights at the following links. For PixArt-alpha, please place the folders for the tokenizer and VAE weights under the same directory.

Model weights: \[[PixArt-alpha](https://huggingface.co/PixArt-alpha/PixArt-alpha/resolve/main/PixArt-XL-2-1024-MS.pth)\, 
                 [PixArt-sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma/blob/main/PixArt-Sigma-XL-2-1024-MS.pth)] <br>
Tokenizer and vae weights: \[PixArt-alpha: ([t5](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl),[vae](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema)), [PixArt-sigma](https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers)\] 

### 0.1 (optional) Precompute the text embedding

> The pixart model family adopts the `T5-XXL` as text encoder, which cost 10GB GPU memory, to save the GPU memory and the long model loading time, we support precompute the text embeddings. 

- `bash ./t2i/shell_scripts/get_text_embeds.sh $GPU_ID`: read the prompts from `./t2i/asset/${TXT_NAME}.txt`, and save the T5 text embeddings into a file named like`text_embeds_alpha_calib`. This file could be specified with the `--precompute_text_embeds` for further processes.
    - We provide 3 prompt lists:
        - `calib.txt`: the first 64 prompts for pixart example prompts.
        - `samples.txt`: the complete 120 prompts for pixart example prompts.
        - `coco_1024.txt`: the first 1024 prompts for coco annotations (used for evaluation). 
    - The text embeds for pixart-alpha and sigma is the same for less than 120 tokens (pixart-alpha maximum token length 120, pixart-sigma maximum token lengyh 300). 

### 0.1 FP16 Inference

- `bash ./t2v/shell_scripts/fp16_inference.sh $GPU_ID`: FP inference for image generation. 
    - configure the `--version` to choose the 'alpha' or 'sigma'
    - specify the path of computed text embeds with `--precompute_text_embeds`

### 1.1 Generate calibration data

- `bash ./t2v/shell_scripts/get_calib_data.sh $GPU_ID`: Generate the calibration data.

### 1.2 Post Training Quantization (PTQ) Process

- `bash ./t2v/shell_scripts/ptq.sh $GPU_ID`: conducting the PTQ process based on calib data, generate the quantized checkpoint.
    - the quantization configs are presented in `t2i/configs/quant/$version` folder, the `w8a8_naive.yaml` is the baseline quantization, and `w8a8.yaml` is the ViDiT-Q plan. 

### 1.3 Quantizad Inference.

- `bash ./t2v/shell_scripts/quant_inference.sh $GPU_ID`: conducting quantized model infernece. 

<br>

# Citation

If you find our work helpful, please consider citing:

```
@misc{zhao2024viditq,
      title={ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation}, 
      author={Tianchen Zhao and Tongcheng Fang and Enshu Liu and Wan Rui and Widyadewi Soedarmadji and Shiyao Li and Zinan Lin and Guohao Dai and Shengen Yan and Huazhong Yang and Xuefei Ning and Yu Wang},
      year={2024},
      eprint={2406.02540},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

# Acknowledgments
Our code was developed based on [opensora v1.0](https://github.com/hpcaitech/Open-Sora)(Apache License), [PixArt-alpha](https://github.com/PixArt-alpha/PixArt-alpha)(AGPL-3.0 license), [PixArt-sigama](https://github.com/PixArt-alpha/PixArt-sigma)(AGPL-3.0 license) and [q-diffusion](https://github.com/Xiuyu-Li/q-diffusion)(MIT License)