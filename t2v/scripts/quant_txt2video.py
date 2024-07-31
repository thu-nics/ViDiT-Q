import os
import sys
# sys.path.append(".")

import torch
import shutil
import logging
from omegaconf import OmegaConf
from mmengine.runner import set_random_seed

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.build_model import build_models
from opensora.utils.misc import to_torch_dtype

from qdiff.models.quant_model import QuantModel
from qdiff.utils import load_quant_params

import inspect
import os

def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False, mode="quant_inference")
    print(cfg)
    PRECOMPUTE_TEXT_EMBEDS = cfg.get('precompute_text_embeds', None)

    opt = cfg
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # INFO: add bakup file and bakup cfg into logpath for debug
    # load the config from the log path
    if not hasattr(opt,"ptq_config"):
        opt.ptq_config = os.path.join(opt.outdir,'config.yaml')
    if not hasattr(opt,"quant_ckpt") or not os.path.exists(opt.quant_ckpt):
        opt.quant_ckpt = os.path.join(opt.outdir,'ckpt.pth')
    if not hasattr(opt,"save_dir"):
        opt.save_dir = os.path.join(opt.outdir,'generated_videos')
    config = OmegaConf.load(f"{opt.ptq_config}")

    log_path = os.path.join(outpath, "quant_inference_run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Conducting Command: %s", " ".join(sys.argv))

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = f"cuda" if torch.cuda.is_available() else "cpu"
    gpus = [int(d) for d in cfg.gpu.split(",")]
    torch.cuda.set_device(gpus[0])
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)
    prompts = load_prompts(cfg.prompt_path)
    prompts = prompts[:cfg.num_videos]

    # ======================================================
    # 3. build model & load weights
    # ======================================================
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


    # scheduler, model, text_encoder, vae, model_args, latent_size = build_models(cfg, device, dtype, enable_sequence_parallelism=False)
    # assert(config.conditional)
    # ======================================================
    # 4. get quantized model
    # ======================================================
    num_timesteps = config.calib_data.n_steps

    assert(config.conditional)

    wq_params = config.quant.weight.quantizer
    aq_params = config.quant.activation.quantizer
    use_weight_quant = True if wq_params else False
    use_act_quant = True if aq_params else False
    if opt.skip_quant_weight:
        use_weight_quant = False
    if opt.skip_quant_act:
        use_act_quant = False

    if config.get('mixed_precision', False):
        if use_weight_quant:
            wq_params['mixed_precision'] = config.mixed_precision
        # if use_act_quant:
        #     aq_params['mixed_precision'] = config.mixed_precision

    qnn = QuantModel(
        model=model, \
        weight_quant_params=wq_params,\
        act_quant_params=aq_params,\
        # act_quant_mode="qdiff",\
        # sm_abit=config.quant.softmax.n_bits,\
    )
    qnn.cuda()
    qnn.eval()
    logger.info(qnn)

    # DIRTY: set the cfg_split as the attribute of the model
    # the cfg_split is configured in `opensora/schedulers/ippdm/__init__.py`
    cfg_split = config.get('cfg_split', False)
    qnn.cfg_split = cfg_split


    qnn.set_quant_state(False, False)
    # for smooth quant
    qnn.set_smooth_quant(smooth_quant=False, smooth_quant_running_stat=False)
    calib_added_cond = {} # It is not required for STDiT

    # with torch.no_grad():
        # if "opensora" in config.model.model_id:
            # _ = qnn(torch.randn(1, 4, 16, 64, 64).cuda(), torch.randint(0, 1000, (1,)).cuda(), torch.randn(1, 1, 120, 4096).cuda(), mask=torch.ones(1, 120).cuda().to(torch.int64))
        # else:
            # raise NotImplementedError

    # for part quantization
    if opt.part_quant:
        quant_layer_list = list(torch.load(config.part_quant_list))
        quant_layer_list = quant_layer_list[:int(len(quant_layer_list) * opt.quant_ratio)]

    if opt.part_fp:
        with open(config.part_fp_list,'r') as f:
            lines = f.readlines()
        fp_layer_list = [line.strip() for line in lines]  # strip the '\n'
        if opt.get('fp_ratio',None) is not None:
            fp_layer_list = fp_layer_list[:int(len(fp_layer_list) * opt.fp_ratio)]
        logger.info("Set the following layers as FP: {}".format(fp_layer_list))

    # for smooth quant
    if aq_params.smooth_quant.enable:
        qnn.set_smooth_quant(smooth_quant=False, smooth_quant_running_stat=False)
        # for i in range(len(sens)):
        #     if sens[i][1]["fp16_diff"] > 7.0 or sens[i][1]["fp16_diff"] < 0.5:
        #         smooth_quant_layer_list.append(sens[i][0])
                # alpha_dict[sens[i][0]] = sens[i][1]["best_alpha"]
        # alpha_dict["model.blocks.27.mlp.fc2"] = 0.675
        qnn.set_smooth_quant(smooth_quant=True, smooth_quant_running_stat=False) # Now we use fp16 to save the statistic of activation
        qnn.set_layer_smooth_quant(model=qnn, module_name_list=fp_layer_list, smooth_quant=False, smooth_quant_running_stat=False)
        # qnn.set_layer_smooth_quant_alpha(model=qnn, alpha_dict=alpha_dict)

    # set the init flag True, otherwise will recalculate params
    if opt.part_quant:
        qnn.set_layer_quant(model=qnn, module_name_list=quant_layer_list, quant_level='per_layer', weight_quant=use_weight_quant, act_quant=use_act_quant, prefix="")
    elif opt.part_fp:
        qnn.set_quant_state(use_weight_quant, use_act_quant)
        qnn.set_layer_quant(model=qnn, module_name_list=fp_layer_list, quant_level='per_layer', weight_quant=False, act_quant=False, prefix="")
    else:
        qnn.set_quant_state(use_weight_quant, use_act_quant) # enable weight quantization, disable act quantization
    qnn.set_quant_init_done('weight')
    qnn.set_quant_init_done('activation')

    load_quant_params(qnn, opt.quant_ckpt)
    qnn.cuda()
    qnn.to(dtype)

    # ======================================================
    # 5. inference
    # ======================================================
    qnn.timestep_wise_quant =False
    sample_idx = 0
    save_dir = opt.save_dir
    os.makedirs(save_dir, exist_ok=True)
    if PRECOMPUTE_TEXT_EMBEDS is not None:
        model_args['precompute_text_embeds'] = torch.load(cfg.precompute_text_embeds)
    for i in range(0, len(prompts), cfg.batch_size):
        batch_prompts = prompts[i : i + cfg.batch_size]
        if PRECOMPUTE_TEXT_EMBEDS is not None:  # also feed in the idxs for saved text_embeds
            model_args['batch_ids'] = torch.arange(i,i+cfg.batch_size)
        samples = scheduler.sample(
            qnn,
            text_encoder,
            sampler_type=cfg.sampler,
            z_size=(vae.out_channels, *latent_size),
            prompts=batch_prompts,
            device=device,
            additional_args=model_args,
        )
        samples = vae.decode(samples.to(dtype))

        for idx, sample in enumerate(samples):
            print(f"Prompt: {batch_prompts[idx]}")
            save_path = os.path.join(save_dir, f"sample_{sample_idx}")
            save_sample(sample, fps=cfg.fps, save_path=save_path)
            sample_idx += 1


if __name__ == "__main__":
    main()
