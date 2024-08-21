EXP_NAME=${2:-"w8a8_naive"}

version="alpha"  # model type (alpha or sigma)
sd_vae_t5="/mnt/public/video_quant/checkpoint/huggingface"  # path to text encoder and vae checkpoints
model_path="./logs/pixart/pixart_alpha/PixArt-XL-2-1024-MS.pth"  # path to PixArt weights
save_path="logs/pixart"  # the path to save generated images
quant_path="logs/pixart/alpha/$EXP_NAME"  # the path of the ptq results
precomputed_text_embeds="./t2i/asset/text_embeds_pixart_alpha.pth"

# # Step 3: Quantized Inference:
CUDA_VISIBLE_DEVICES=$1 python ./t2i/scripts/calibrate_ptqd_k.py \
        --version $version \
        --pipeline_load_from $sd_vae_t5 \
        --model_path $model_path \
        --quant_path  $quant_path \
        --save_path $save_path \
        --quant_act \
        --quant_weight \
        --precomputed_text_embeds $precomputed_text_embeds \
