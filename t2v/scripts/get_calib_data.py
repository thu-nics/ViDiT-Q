import os
import sys
# sys.path.append(".")

import torch
import colossalai
import torch.distributed as dist
from mmengine.runner import set_random_seed

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from colossalai.cluster import DistCoordinator


def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False, mode="get_calib")
    print(cfg)

    # init distributed
    # colossalai.launch_from_torch({})
    # coordinator = DistCoordinator()

    # if coordinator.world_size > 1:
        # set_sequence_parallel_group(dist.group.WORLD) 
        # enable_sequence_parallelism = True
    # else:
        # enable_sequence_parallelism = False
    
    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)
    prompts = load_prompts(cfg.prompt_path)
    prompts = prompts[:cfg.data_num]
    PRECOMPUTE_TEXT_EMBEDS = cfg.get('precompute_text_embeds', None)

    # ======================================================
    # 3. build model & load weights
    # =====================================
    # 
    # =================
    # 3.1. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    # 3.2. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        # caption_channels=text_encoder.output_dim,
        caption_channels=4096,  # DIRTY: for T5 only
        model_max_length=cfg.text_encoder.model_max_length,
        dtype=dtype,
        enable_sequence_parallelism=False,
    )
    if PRECOMPUTE_TEXT_EMBEDS is not None:
        text_encoder = None
    else:
        text_encoder = build_module(cfg.text_encoder, MODELS, device=device)  # T5 must be fp32
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
        model_args["data_info"] = dict(ar=ar, hw=hw)

    # ======================================================
    # 4. inference
    # ======================================================
    sample_idx = 0
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    calib_data = {}
    input_data_list = []
    output_data_list = []
    if PRECOMPUTE_TEXT_EMBEDS is not None:
        model_args['precompute_text_embeds'] = torch.load(cfg.precompute_text_embeds)

    for i in range(0, len(prompts), cfg.batch_size):
        batch_prompts = prompts[i : i + cfg.batch_size]
        if PRECOMPUTE_TEXT_EMBEDS is not None:  # also feed in the idxs for saved text_embeds
            model_args['batch_ids'] = torch.arange(i,i+cfg.batch_size)
        samples, cur_calib_data, out_data = scheduler.sample(
            model,
            text_encoder,
            sampler_type=cfg.sampler,
            z_size=(vae.out_channels, *latent_size),
            prompts=batch_prompts,
            device=device,
            return_trajectory=True,
            additional_args=model_args,
        )

        for key in cur_calib_data:
            if not key in calib_data.keys():
                calib_data[key] = cur_calib_data[key]
            else:
                calib_data[key] = torch.cat([cur_calib_data[key], calib_data[key]], dim=1)

        input_data_list.append(cur_calib_data)
        output_data_list.append(out_data)
        # save samples
        samples = vae.decode(samples.to(dtype))
        # if coordinator.is_master():
        for idx, sample in enumerate(samples):
            print(f"Prompt: {batch_prompts[idx]}")
            save_path = os.path.join(save_dir, f"sample_{sample_idx}")
            save_sample(sample, fps=cfg.fps, save_path=save_path)
            sample_idx += 1

    # ======================================================
    # 5. save calibration data
    # ======================================================
    torch.save(calib_data, os.path.join(save_dir, "calib_data.pt"))
    if cfg.save_inp_oup:
        torch.save(input_data_list, os.path.join(save_dir, "input_list.pt"))
        torch.save(output_data_list, os.path.join(save_dir, "output_list.pt"))


if __name__ == "__main__":
    main()
