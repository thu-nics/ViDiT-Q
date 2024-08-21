version="alpha"  # model type (alpha or sigma)
sd_vae_t5="/mnt/public/video_quant/checkpoint/huggingface"  # path to text encoder and vae checkpoints
model_path="./logs/pixart/pixart_alpha/PixArt-XL-2-1024-MS.pth"  # path to PixArt weights
txt_file_name="samples"   # ['calib'. 'samples', 'coco_1024']
txt_file_path="./t2i/asset/$txt_file_name.txt"
save_path="./t2i/asset/"  # not used

# (optional) Step 0: precompute the text embeds to avoid loading the large T5 model.
CUDA_VISIBLE_DEVICES=$1 python ./t2i/scripts/get_calib_data.py \
	--version $version \
	--txt_file $txt_file_path \
	--pipeline_load_from $sd_vae_t5 \
	--model_path $model_path \
	--save_path $save_path \
	--save_text_embeds_only \
	--bs 8 \
