EXP_NAME=${2:-"w8a8_naive"}

version="alpha"  # model type (alpha or sigma)
sd_vae_t5="/mnt/public/video_quant/checkpoint/huggingface"  # path to text and image encoder checkpoints
model_path="./logs/pixart/pixart_alpha/PixArt-XL-2-1024-MS.pth"  # path to PixArt weights
exp_name=$EXP_NAME  # quantization bit width [w8a8, w4a8]
save_path="./logs/pixart"  # the path to save the result
#ptq_config="t2i/configs/quant/alpha/pixart-dpm_w8a8.yml"  # the quantization config
ptq_config="t2i/configs/quant/alpha/$EXP_NAME.yaml"  # the quantization config
calib_data_path="./logs/pixart/calib_data"
precomputed_text_embeds="./t2i/asset/text_embeds_pixart_alpha.pth"

# Step 2: Post-Training Quantization:
CUDA_VISIBLE_DEVICES=$1 python ./t2i/scripts/ptq.py \
        --version $version \
        --pipeline_load_from $sd_vae_t5 \
        --model_path $model_path \
        --exp_name $exp_name \
        --save_path $save_path \
        --ptq_config $ptq_config \
        --calib_data_path $calib_data_path \
        --precomputed_text_embeds $precomputed_text_embeds \
