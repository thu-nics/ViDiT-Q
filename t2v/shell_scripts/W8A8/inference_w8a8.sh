CFG="./t2v/configs/quant/W8A8/16x512x512.py" # the opensora config
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"  # your path of splited ckpt
OUTDIR="./logs/w8a8_ptq"  # your path of the w8a8 ptq result
GPU_ID=$1
SAVE_DIR="w8a8_samples"  # your path to save generated videos

# quant inference
CUDA_VISIBLE_DEVICES=$GPU_ID python t2v/scripts/quant_txt2video.py $CFG \
    --outdir $OUTDIR --ckpt_path $CKPT_PATH  \
    --dataset_type opensora \
    --part_fp \
    --save_dir $SAVE_DIR \
    --precompute_text_embeds ./t2v/utils_files/text_embeds.pth
