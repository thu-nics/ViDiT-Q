version=$1  # model type (alpha or sigma)
sd_vae=$2  # path to text and image encoder checkpoints
model_path=$3  # path to PixArt weights
ptq_config=$4  # quantization bit width [w8a8, w4a8]
save_path="./logs/pixart/calib_data"

# Step 1: Obtaining the Calibration Dataset:
python ./t2i/scripts/get_calib_data.py \
        --version "$version" \
        --pipeline_load_from "$sd_vae" \
        --model_path "$model_path" \
        --save_path "$save_path"

# Step 2: Post-Training Quantization:
# python ./t2i/scripts/ptq.py \
#         --pipeline_load_from "$sd_vae" \
#         --model_path "$model_path" \
#         --ptq_config "$ptq_config" \

# # Step 3: Quantized Inference:
# python ./t2i/scripts/quant_txt2img.py \
#         --version "$version" \
#         --pipeline_load_from "$sd_vae" \
#         --model_path "$model_path" \
#         --ptq_config "$ptq_config" \
#         --quant_act True \
#         --quant_weight True