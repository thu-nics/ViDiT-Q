version="sigma"  # model type (alpha or sigma)
sd_vae_t5="/mnt/public/video_quant/checkpoint/pixart/PixArt-sigma/sdxl-vae_t5"  # path to text encoder and vae checkpoints
model_path="./logs/pixart/pixart_sigma/PixArt-Sigma-XL-2-1024-MS.pth"  # path to PixArt weights
bitwidth_setting="w8a8"  # quantization bit width [w8a8, w4a8]
save_path="./logs/pixart/calib_data"  # the path to save calibration dataset
GPU_ID=$1

# Step 1: Obtaining the Calibration Dataset:
python ./t2i/scripts/get_calib_data.py \
        --version $version \
        --pipeline_load_from $sd_vae_t5 \
        --model_path $model_path \
        --save_path $save_path \
        --gpu $GPU_ID
