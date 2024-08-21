EXP_NAME=${2:-"fp16"}

version="sigma"  # ['alpha','sigma']
if [ "$version" == "alpha" ]; then
	sd_vae_t5="/mnt/public/video_quant/checkpoint/huggingface"  # path to text encoder and vae checkpoints
	model_path="./logs/pixart/pixart_alpha/PixArt-XL-2-1024-MS.pth"  # path to PixArt weights
elif [ "$version" == "sigma" ]; then
	sd_vae_t5="/mnt/public/video_quant/checkpoint/pixart/PixArt-sigma/sdxl-vae_t5"  # path to text encoder and vae checkpoints
	model_path="./logs/pixart/pixart_sigma/PixArt-Sigma-XL-2-1024-MS.pth"  # path to PixArt weights
else
	echo "wrong model version, should be 'alpha' or 'sigma'."
fi

save_path="logs/pixart"  # the path to save generated images
exp_name=$EXP_NAME
txt_file_name="calib"   # ['calib'. 'samples', 'coco_1024']
txt_file_path="./t2i/asset/$txt_file_name.txt"
calib_data_path="./logs/pixart/calib_data"
precomputed_text_embeds="./t2i/asset/text_embeds_${version}_${txt_file_name}.pth"  # optional, if generated


# # Step 3: Quantized Inference:
CUDA_VISIBLE_DEVICES=$1 python ./t2i/scripts/inference.py \
        --version $version \
		--txt_file $txt_file_path \
        --pipeline_load_from $sd_vae_t5 \
        --model_path $model_path \
		--exp_name $exp_name \
        --save_path $save_path \
		--bs 8 \
		--precomputed_text_embeds $precomputed_text_embeds \
