CFG="./t2v/configs/opensora/inference/16x512x512.py"  # the opensora config
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"  # your path of splited ckpt
OUTDIR="./logs/fp16_inference"  # your_path_to_save_videos
GPU_ID=$1

CUDA_VISIBLE_DEVICES=$GPU_ID python t2v/scripts/inference.py $CFG --ckpt_path $CKPT_PATH  --outdir $OUTDIR \
--precompute_text_embeds ./t2v/utils_files/text_embeds.pth