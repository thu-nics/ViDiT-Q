import torch

from opensora.registry import MODELS, SCHEDULERS, build_module

def build_models(cfg, device, dtype, enable_sequence_parallelism):
    # ======================================================
    # build model & load weights
    # =====================================
    # 
    # =================
    # 1. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    # 2. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)  # T5 must be fp32
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        dtype=dtype,
        enable_sequence_parallelism=enable_sequence_parallelism,
    )
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 3. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    # 4. support for multi-resolution
    model_args = dict()
    if cfg.multi_resolution:
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)
        
    return scheduler, model, text_encoder, vae, model_args, latent_size

def build_models_only(cfg, device, dtype, enable_sequence_parallelism, vae_latent_size=[16, 64, 64], vae_out_channels=4, text_out_dim=4096, text_max_length=120):
    model = build_module(
        cfg.model,
        MODELS,
        input_size=vae_latent_size,
        in_channels=vae_out_channels,
        caption_channels=text_out_dim,
        model_max_length=text_max_length,
        dtype=dtype,
        enable_sequence_parallelism=enable_sequence_parallelism,
    )
    model = model.to(device, dtype).eval()
    
    return model
