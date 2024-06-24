import os
import sys
# sys.path.append(".")

import torch
from mmengine.runner import set_random_seed
from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype

import inspect

def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts

def main():
    # 1. cfg
    cfg = parse_configs(training=False)
    print(cfg)

    # 2. runtime variables
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)
    prompts = load_prompts(cfg.prompt_path)

    # 3. build model & load weights
    # 3.1. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    # 3.2. build model
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
    )
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 3.3. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    # 3.4. support for multi-resolution
    model_args = dict()
    if cfg.multi_resolution:
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        # Assume model_args is a dictionary that you might need to pass to the model
        model_args["data_info"] = dict(ar=ar, hw=hw)

    # 4. inference
    sample_idx = 0
    outdir = cfg.outdir
    os.makedirs(outdir, exist_ok=True)

    text_embeds = []
    masks = []
    for i in range(0, len(prompts), cfg.batch_size):
        batch_prompts = prompts[i : i + cfg.batch_size]
        n = len(batch_prompts)
        model_args = text_encoder.encode(batch_prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.stack([model_args["y"], y_null], 1)
        text_embeds.append(model_args["y"])
        # since cond and uncond, the y have dim of 2, so do mask
        masks.append(model_args["mask"])

    text_embeds = torch.cat(text_embeds, dim=0)
    masks = torch.cat(masks,dim=0)
    d = {
            'y': text_embeds,
            'mask': masks,
            }
    torch.save(d, os.path.join(outdir,"text_embeds.pth"))



if __name__ == "__main__":
    main()
