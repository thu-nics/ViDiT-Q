EXP_NAME=${2:-"w8a8_naive"}

version="alpha"  # model type (alpha or sigma)
sd_vae_t5="/mnt/public/video_quant/checkpoint/huggingface"  # path to text encoder and vae checkpoints
model_path="./logs/pixart/pixart_alpha/PixArt-XL-2-1024-MS.pth"  # path to PixArt weights
save_path="logs/pixart"  # the path to save generated images
quant_path="logs/pixart/alpha/$EXP_NAME"  # the path of the ptq results
GPU_ID=$1

# # Step 3: Quantized Inference:
python ./t2i/scripts/quant_txt2img.py \
        --version $version \
        --pipeline_load_from $sd_vae_t5 \
        --model_path $model_path \
        --quant_path  $quant_path \
        --save_path $save_path \
        --quant_act \
        --quant_weight \
        --gpu $GPU_ID
