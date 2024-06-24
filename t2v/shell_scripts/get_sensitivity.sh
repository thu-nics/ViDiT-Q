# get the sensitivity through the sensitivity
CFG="./t2v/configs/quant/W8A8/16x512x512.py" # the opensora config
CKPT_PATH="./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"  # splited ckpt
TEXT_EMBEDS_OUTDIR="./logs/text_embeds/opensora"
OUTDIR="./logs/w8a8_ptq"  # your path of the w8a8 ptq result
GPU_ID=$1
PRE_COMPUTED_TEXTEMB="./t2v/utils_files/text_embeds.pth"
quant_group='.attn.'  # we split the model in to 4 groups: ['.attn.', 'attn_temp', 'cross_attn', 'mlp']
SAVE_PATH="w8a8_sensitivity_$quant_group"  # your path to save generated videos


# # group wise quant
# python t2v/scripts/get_sensitivity.py $CFG --ckpt_path $CKPT_PATH --outdir $OUTDIR --save_dir $SAVE_PATH --dataset_type opensora --precompute_text_embeds $PRE_COMPUTED_TEXTEMB --part_fp \
# --group_wise_quant

# # layer wise quant
# python t2v/scripts/get_sensitivity.py $CFG --ckpt_path $CKPT_PATH --outdir $OUTDIR --save_dir $SAVE_PATH --dataset_type opensora --precompute_text_embeds $PRE_COMPUTED_TEXTEMB --part_fp \
# --layer_wise_quant

# # timestep wise quant
# python t2v/scripts/get_sensitivity.py $CFG --ckpt_path $CKPT_PATH --outdir $OUTDIR --save_dir $SAVE_PATH --dataset_type opensora --precompute_text_embeds $PRE_COMPUTED_TEXTEMB --part_fp \
# --timestep_wise_quant

# # timestep wise quant + layer wise quant
# python t2v/scripts/get_sensitivity.py $CFG --ckpt_path $CKPT_PATH --outdir $OUTDIR --save_dir $SAVE_PATH --dataset_type opensora --precompute_text_embeds $PRE_COMPUTED_TEXTEMB --part_fp \
# --group_wise_quant --timestep_wise_quant

# # timestep wise quant + group wise quant
# python t2v/scripts/get_sensitivity.py $CFG --ckpt_path $CKPT_PATH --outdir $OUTDIR --save_dir $SAVE_PATH --dataset_type opensora --precompute_text_embeds $PRE_COMPUTED_TEXTEMB --part_fp \
# --layer_wise_quant --timestep_wise_quant

# timestep wise quant + block wise quant + group wise quant
python t2v/scripts/get_sensitivity.py $CFG --ckpt_path $CKPT_PATH --outdir $OUTDIR --save_dir $SAVE_PATH --dataset_type opensora --precompute_text_embeds $PRE_COMPUTED_TEXTEMB --part_fp \
--block_group_wise_quant --timestep_wise_quant --group_quant group_quant