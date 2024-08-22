CFG="./t2v/configs/opensora/inference/16x512x512.py" # the opensora config
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"  # splited ckpt
GPU_ID=$1
CALIB_DATA_DIR="./logs/calib_data"  # the path to save your calib dataset

## quant calib data
CUDA_VISIBLE_DEVICES=$GPU_ID python t2v/scripts/get_calib_data.py $CFG --ckpt_path $CKPT_PATH --data_num 10 --outdir $CALIB_DATA_DIR --save_dir $CALIB_DATA_DIR \
--precompute_text_embeds ./t2v/utils_files/text_embeds.pth