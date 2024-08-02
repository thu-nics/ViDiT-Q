import os
import sys
# sys.path.append(".")

import argparse, datetime, gc, yaml
import logging
import numpy as np
from tqdm import tqdm, trange
from omegaconf import OmegaConf
import torch
import shutil
from mmengine.runner import set_random_seed
from pytorch_lightning import seed_everything

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.build_model import build_models
from opensora.utils.misc import to_torch_dtype

from qdiff.models.quant_model import QuantModel
from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer
from qdiff.utils import get_quant_calib_data


logger = logging.getLogger(__name__)

def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts

def main():

    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False, mode="ptq")
    print(cfg)
    PRECOMPUTE_TEXT_EMBEDS = cfg.get('precompute_text_embeds', None)

    opt = cfg
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # INFO: add bakup file and bakup cfg into logpath for debug
    if os.path.exists(os.path.join(outpath,'config.yaml')):
        os.remove(os.path.join(outpath,'config.yaml'))
    shutil.copy(opt.ptq_config, os.path.join(outpath,'config.yaml'))
    shutil.copy(opt.config, os.path.join(outpath,'opensora_config.py'))
    if os.path.exists(os.path.join(outpath,'qdiff')): # if exist, overwrite
        shutil.rmtree(os.path.join(outpath,'qdiff'))
    shutil.copytree('./qdiff', os.path.join(outpath,'qdiff'))

    log_path = os.path.join(outpath, "run.log")
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
    config = OmegaConf.load(f"{opt.ptq_config}")
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
    init_noise = torch.randn(len(prompts), *(vae.out_channels, *latent_size))
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

    # if PRECOMPUTE_TEXT_EMBEDS:
        # del cfg.text_encoder
        # text_encoder = None
        # scheduler, model, vae, model_args, latent_size = build_models(cfg, device, dtype, enable_sequence_parallelism=False)
        # import ipdb; ipdb.set_trace()
    # else:
        # scheduler, model, text_encoder, vae, model_args, latent_size = build_models(cfg, device, dtype, enable_sequence_parallelism=False)
        # assert(config.conditional)

    # ======================================================
    # 4. build quantized model
    # ======================================================
    # TODO: only feed part of the wq_params, since it is directly used for quantizer init
    wq_params = config.quant.weight.quantizer
    aq_params = config.quant.activation.quantizer
    # use_weight_quant = False if wq_params is None else True
    # use_act_quant = False if aq_params is None else True

    if config.get('mixed_precision', False):
        wq_params['mixed_precision'] = config.mixed_precision
        # aq_params['mixed_precision'] = config.mixed_precision
    if config.get('timestep_wise', False):
        aq_params['timestep_wise'] = config.timestep_wise
        assert config.calib_data.n_steps == cfg.scheduler.num_sampling_steps, "When PTQ, the scheduler steps should equal to timesteps to record"

    qnn = QuantModel(
        model=model, \
        weight_quant_params=wq_params,\
        act_quant_params=aq_params,\
    )
    qnn.cuda()
    qnn.eval()
    logger.info(qnn)

    if not config.quant.grad_checkpoint:
        logger.info('Not use gradient checkpointing for transformer blocks')
        qnn.set_grad_ckpt(False)

    logger.info(f"Sampling data from {config.calib_data.n_steps} timesteps for calibration")
    # INFO: feed the calib_data path through argparse also, overwrite quant config
    if hasattr(cfg,"calib_data"):
        if cfg.calib_data is not None:
            config.calib_data.path = cfg.calib_data
    calib_data_ckpt = torch.load(config.calib_data.path, map_location='cpu')
    calib_data = get_quant_calib_data(config, calib_data_ckpt, config.calib_data.n_steps, model_type=config.model.model_type, repeat_interleave=cfg.get('timestep_wise',False))
    del(calib_data_ckpt)
    gc.collect()

    # ======================================================
    # 5. prepare data for init the model
    # ======================================================
    calib_added_kwargs = {}
    if config.model.model_type == "opensora":
        calib_batch_size = config.calib_data.batch_size*2  # INFO: used to support the CFG
        logger.info(f"Calibration data shape: {calib_data[0].shape} {calib_data[1].shape} {calib_data[2].shape}")
        calib_xs, calib_ts, calib_cs, calib_masks = calib_data
        calib_added_kwargs["mask"] = calib_masks
    else:
        raise NotImplementedError

    # ======================================================
    # 6. get the quant params (training-free), using the calibration data
    # ======================================================

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

    with torch.no_grad():
        # --- get temp kwargs -----
        if config.model.model_type == "opensora":
            # original model takes 2*bs 'y' and bs mask
            tmp_kwargs = {"mask": calib_added_kwargs["mask"][:calib_batch_size][::2].cuda()}
        else:
            tmp_kwargs = calib_added_kwargs

        qnn.set_module_name_for_quantizer(module=qnn.model)  # add the module name as attribute for each quantizer
        # _ = qnn(calib_xs[:calib_batch_size].cuda(), calib_ts[:calib_batch_size].cuda(), calib_cs[:calib_batch_size].cuda(), **tmp_kwargs)
        
        if aq_params.smooth_quant.enable: 
            assert aq_params.get('dynamic',False)
            qnn.set_smooth_quant(smooth_quant=True, smooth_quant_running_stat=False) # Now we use fp16 to save the statistic of activation
            
        # calculte the running acitvation for smooth quant
        calib_xs_save = calib_xs
        calib_ts_save = calib_ts
        calib_cs_save = calib_cs
        if aq_params.smooth_quant.enable: 
            logger.info("begin to calculate the statistic of activation for smooth quant!")
            assert aq_params.get('dynamic',False)
            qnn.set_smooth_quant(smooth_quant=False, smooth_quant_running_stat=True) # Now we use fp16 to save the statistic of activation
            qnn.set_quant_state(False, False)
            # The following code is the same as activation quantization, to calculate the statistic
            # Need to support different timestep, calib activation respectively, we have to split the time_steps
            calib_n_samples = config.calib_data.n_samples*2
            calib_ts = calib_ts.reshape([-1,calib_n_samples])
            # INFO: when the calib_n_samples is smaller than calib data timestep size
            # e.g., 100 / 1000, the result would be [100,10] -> [990,...x10],[980,...,x10]
            calib_n_steps = calib_ts.shape[0]
            calib_xs = calib_xs.reshape([calib_n_steps,calib_n_samples]+list(calib_xs.shape[1:])) # split the 1st dim (batch) into 2
            calib_cs = calib_cs.reshape([calib_n_steps,calib_n_samples]+list(calib_cs.shape[1:]))
            calib_masks = calib_masks.reshape([calib_n_steps,calib_n_samples]+list(calib_masks.shape[1:]))

            inds = np.arange(calib_xs.shape[1])
            np.random.shuffle(inds)
            rounds = int(calib_xs.size(1) / calib_batch_size)

            for i_ts in trange(calib_n_steps):
                assert torch.all(calib_ts[i_ts,:] == calib_ts[i_ts,0])  # ts have the same timestepe_id
                # qnn.set_timestep_for_quantizer(calib_ts[i_ts,0].item())
                for i in range(rounds):
                    if config.model.model_type == "opensora":
                        _ = qnn(\
                            calib_xs[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                            calib_ts[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),\
                            calib_cs[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                            mask=calib_masks[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                        )
                    else:
                        raise NotImplementedError
            qnn.set_smooth_quant(smooth_quant=True, smooth_quant_running_stat=False)
            qnn.set_layer_smooth_quant(model=qnn, module_name_list=fp_layer_list, smooth_quant=False, smooth_quant_running_stat=False)
        calib_xs = calib_xs_save
        calib_ts = calib_ts_save
        calib_cs = calib_cs_save

        logger.info("smooth quant running activation calculate done!")
        torch.cuda.empty_cache()
        
        alpha_list = np.arange(0.475, 0.9, 0.01).tolist()
        ori_outpath = outpath
        
        for alpha in alpha_list:
            torch.manual_seed(cfg.seed)
            qnn.set_quant_init_undone('weight')
            qnn.set_quant_init_undone('activation')
            # --------------- Set Global Smooth Quant Alpha ---------------
            outpath = os.path.join(ori_outpath, "alpha_{:.4f}".format(alpha))
            os.makedirs(outpath, exist_ok=True)
            alpha_dict = {"": alpha}
            qnn.set_layer_smooth_quant_alpha(model=qnn, alpha_dict=alpha_dict, prefix="")
            # --------------- Quantization Part ---------------
            # --- the weight quantization -----
            # enable part quantization
            if opt.part_quant:
                qnn.set_layer_quant(model=qnn, module_name_list=quant_layer_list, quant_level='per_layer', weight_quant=True, act_quant=False, prefix="")
            elif opt.part_fp:
                qnn.set_quant_state(True, False)
                qnn.set_layer_quant(model=qnn, module_name_list=fp_layer_list, quant_level='per_layer', weight_quant=False, act_quant=False, prefix="")
            else:
                qnn.set_quant_state(True, False) # enable weight quantization, disable act quantization
                
            _ = qnn(calib_xs[:calib_batch_size].cuda(), calib_ts[:calib_batch_size].cuda(), calib_cs[:calib_batch_size].cuda(), **tmp_kwargs)
            logger.info("weight initialization done!")
            qnn.set_quant_init_done('weight')
            torch.cuda.empty_cache()

            # --- the activation quantization -----
            # by default, use the running_mean of calibration data to determine activation quant params
            if opt.part_quant:
                qnn.set_layer_quant(model=qnn, module_name_list=quant_layer_list, quant_level='per_layer', weight_quant=True, act_quant=True, prefix="")
            elif opt.part_fp:
                qnn.set_quant_state(True, True)
                qnn.set_layer_quant(model=qnn, module_name_list=fp_layer_list, quant_level='per_layer', weight_quant=False, act_quant=False, prefix="")
            else:
                qnn.set_quant_state(True, True) # quantize activation with fixed quantized weight
            logger.info('Running stat for activation quantization')


            if aq_params.get('dynamic',False):
                logger.info('Adopting dynamic quant params, skip calculating fixed quant params')
            else:
                if not config.get('timestep_wise', False):
                    # Normal activation calibration, walk through all calib data
                    inds = np.arange(calib_xs.shape[0])
                    # np.random.shuffle(inds)  # ERROR: using shuffle would make it mixed
                    rounds = int(calib_xs.size(0) / calib_batch_size)

                    for i in trange(rounds):
                        if config.model.model_type == "opensora":
                            _ = qnn(calib_xs[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),
                                calib_ts[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),
                                calib_cs[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),
                                mask=calib_masks[inds[i * calib_batch_size:(i + 1) * calib_batch_size]][::2].cuda(),  # INFO: original opensora model takes in 2*bs conds(y) and bs mask
                            )
                        else:
                            raise NotImplementedError
                else:
                    # Need to support different timestep, calib activation respectively, we have to split the time_steps
                    calib_n_samples = config.calib_data.n_samples*2
                    calib_ts = calib_ts.reshape([-1,calib_n_samples])
                    # INFO: when the calib_n_samples is smaller than calib data timestep size
                    # e.g., 100 / 1000, the result would be [100,10] -> [990,...x10],[980,...,x10]
                    calib_n_steps = calib_ts.shape[0]
                    calib_xs = calib_xs.reshape([calib_n_steps,calib_n_samples]+list(calib_xs.shape[1:])) # split the 1st dim (batch) into 2
                    calib_cs = calib_cs.reshape([calib_n_steps,calib_n_samples]+list(calib_cs.shape[1:]))
                    calib_masks = calib_masks.reshape([calib_n_steps,calib_n_samples]+list(calib_masks.shape[1:]))

                    inds = np.arange(calib_xs.shape[1])
                    np.random.shuffle(inds)
                    rounds = int(calib_xs.size(1) / calib_batch_size)

                    for i_ts in trange(calib_n_steps):
                        assert torch.all(calib_ts[i_ts,:] == calib_ts[i_ts,0])  # ts have the same timestepe_id
                        qnn.set_timestep_for_quantizer(calib_ts[i_ts,0].item())
                        for i in range(rounds):
                            if config.model.model_type == "opensora":
                                _ = qnn(\
                                        calib_xs[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                                        calib_ts[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),\
                                        calib_cs[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                                        mask=calib_masks[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                                )
                            else:
                                raise NotImplementedError

                    # INFO: re-fill the 1000 timesteps quant params if timestep-wise 
                    if config.get('timestep_wise', False):
                        qnn.repeat_timestep_wise_quant_params(calib_ts)

            qnn.set_quant_init_done('activation')
            # save the quant params
            logger.info("Saving calibrated quantized DiT model")
            quant_params_dict = qnn.get_quant_params_dict()
            torch.save(quant_params_dict, os.path.join(outpath, "ckpt.pth"))
            
            
            # --------------- Quant Inference Part ---------------
            if opt.skip_quant_weight:
                use_weight_quant = False
            else:
                use_weight_quant = True
            if opt.skip_quant_act:
                use_act_quant = False
            else:
                use_act_quant = True
            # qnn.set_quant_state(use_weight_quant, use_act_quant)
            if opt.part_quant:
                qnn.set_layer_quant(model=qnn, module_name_list=quant_layer_list, quant_level='per_layer', weight_quant=use_weight_quant, act_quant=use_act_quant, prefix="")
            elif opt.part_fp:
                qnn.set_quant_state(use_weight_quant, use_act_quant)
                qnn.set_layer_quant(model=qnn, module_name_list=fp_layer_list, quant_level='per_layer', weight_quant=False, act_quant=False, prefix="")
            else:
                qnn.set_quant_state(use_weight_quant, use_act_quant) # enable weight quantization, disable act quantization
                
            qnn.timestep_wise_quant =False
            sample_idx = 0
            save_dir = os.path.join(outpath, "videos")
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
                    init_noise=init_noise[:len(batch_prompts)]
                )
                samples = vae.decode(samples.to(dtype))

                for idx, sample in enumerate(samples):
                    print(f"Prompt: {batch_prompts[idx]}")
                    save_path = os.path.join(save_dir, f"sample_{sample_idx}")
                    save_sample(sample, fps=cfg.fps, save_path=save_path)
                    sample_idx += 1

if __name__ == "__main__":
    main()