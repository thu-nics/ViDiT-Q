version="sigma"  # model type (alpha or sigma)
sd_vae_t5="/mnt/public/video_quant/checkpoint/pixart/PixArt-sigma/sdxl-vae_t5"  # path to text and image encoder checkpoints
model_path="./logs/pixart/pixart_sigma/PixArt-Sigma-XL-2-1024-MS.pth"  # path to PixArt weights
bitwidth_setting="w8a8"  # quantization bit width [w8a8, w4a8]
save_path="./logs/pixart"  # the path to save the result of the ptq
ptq_config="t2i/configs/quant/sigma/pixart-dpm_w8a8.yml"  # the quantization config
calib_data_path="./logs/pixart/calib_data"
GPU_ID=$1

# Step 2: Post-Training Quantization:
python ./t2i/scripts/ptq.py \
        --version $version \
        --pipeline_load_from $sd_vae_t5 \
        --model_path $model_path \
        --bitwidth_setting $bitwidth_setting \
        --save_path $save_path \
        --ptq_config $ptq_config \
        --calib_data_path $calib_data_path \
        --gpu $GPU_ID