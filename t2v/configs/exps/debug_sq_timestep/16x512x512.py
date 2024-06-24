num_frames = 16
fps = 24 // 3
image_size = (512, 512)

# Define model
model = dict(
    type="STDiT-XL/2",
    space_scale=1.0,
    time_scale=1.0,
    enable_flashattn=True, # default is True
    enable_layernorm_kernel=False, # default is True
    from_pretrained="PRETRAINED_MODEL"
)
vae = dict(
    type="VideoAutoencoderKL",
    from_pretrained="/share/liuenshu/temp_files/checkpoints/huggingface/sd-vae-ft-ema",
    micro_batch_size=128,
)
text_encoder = dict(
    type="t5",
    from_pretrained="./pretrained_models/t5_ckpts",
    save_pretrained="/share/liuenshu/temp_files/checkpoints/text_encoder/pretrained_t5",
    model_max_length=120,
)
scheduler = dict(
    type="iddpm",
    num_sampling_steps=20,
    cfg_scale=7.0,
)
dtype = "fp16"

# Others
# batch_size = 2
# seed = 42
# prompt_path = "/share/public/video_quant/wanrui/datasets/ucf101/videos/prompt_ucf.txt"
# save_dir = "/share/liuenshu/temp_files/video_quant/inp_oup_data.pth"
batch_size = 4
seed = 42
prompt_path = "./t2v/assets/texts/t2v_samples.txt"
# save_dir = "./generated_videos/fp16"

