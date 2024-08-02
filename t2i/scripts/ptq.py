import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tqdm import tqdm, trange
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer

import sys
# sys.path.append(".")

import gc, yaml
import numpy as np
import logging
import shutil
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from diffusion.model.utils import prepare_prompt_ar
from diffusion import IDDPM, DPMS_alpha, DPMS_sigma, SASolverSampler
from tools.download import find_model
from diffusion.model.t5 import T5Embedder
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from diffusion.data.datasets import get_chunks
from diffusion.data.datasets.utils import *

from qdiff.models.quant_model import QuantModel
from qdiff.quantizer.base_quantizer import BaseQuantizer, WeightQuantizer, ActQuantizer
from qdiff.utils import get_quant_calib_data
from qdiff.optimization.model_recon import model_reconstruction

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--version', default='sigma', type=str)
    parser.add_argument(
        "--pipeline_load_from", default='output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument('--txt_file', default='./t2i/asset/samples.txt', type=str)
    parser.add_argument('--model_path', default='output/pretrained_models/PixArt-XL-2-1024x1024.pth', type=str)
    parser.add_argument('--sdvae', action='store_true', help='sd vae')
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default='custom', type=str)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--save_name', default='test_sample', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--ptq_config', default='./t2i/configs/quant/', type=str)
    parser.add_argument('--exp_name', default='', type=str)
    parser.add_argument('--save_path', required=True, type=str)
    parser.add_argument('--calib_data_path', required=True, type=str, default=None)
    parser.add_argument('--base', default=False, type=bool)
    parser.add_argument('--precomputed_text_embeds', default=None, type=str)
    # parser.add_argument('--smoothq', default=False, type=bool, help="if to perform smooth quant")

    return parser.parse_args()

def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)

def visualize(items, bs, sample_steps, cfg_scale):

    for chunk in tqdm(list(get_chunks(items, bs))[0], unit='batch'):

        prompts = []
        if bs == 1:
            prompt_clean, _, hw, ar, custom_hw = prepare_prompt_ar(chunk[0], base_ratios, device=device, show=False)  # ar for aspect ratio
            if args.image_size == 1024:
                latent_size_h, latent_size_w = int(hw[0, 0] // 8), int(hw[0, 1] // 8)
            else:
                hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
                ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
                latent_size_h, latent_size_w = latent_size, latent_size
            prompts.append(prompt_clean.strip())
        else:
            hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device).repeat(bs, 1)
            ar = torch.tensor([[1.]], device=device).repeat(bs, 1)
            for prompt in chunk:
                prompts.append(prepare_prompt_ar(prompt, base_ratios, device=device, show=False)[0].strip())
            latent_size_h, latent_size_w = latent_size, latent_size
        if args.version == 'sigma':
            caption_token = tokenizer(prompts, max_length=max_sequence_length, padding="max_length", truncation=True,
                                    return_tensors="pt").to(device)
            caption_embs = text_encoder(caption_token.input_ids, attention_mask=caption_token.attention_mask)[0]
            caption_embs = caption_embs[:, None]
            emb_masks = caption_token.attention_mask
            null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]
        elif args.version == 'alpha':
            if args.precomputed_text_embeds is not None:
                assert args.txt_file == './t2i/asset/samples.txt'
                save_d = torch.load('./t2i/asset/text_embeds_pixart_alpha.pth')
                caption_embs = save_d['caption_embs']
                emb_masks = save_d['emb_masks']
                null_y = save_d['null_y']
            else:
                caption_embs, emb_masks = t5.get_text_embeddings(prompts)
                caption_embs = caption_embs.float()[:,None]
                null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]

                # INFO: save the text_embeds
                # save_d = {}
                # save_d['caption_embs'] = caption_embs
                # save_d['emb_masks'] = emb_masks
                # save_d['null_y'] = null_y
                # torch.save(save_d, './t2i/asset/text_embeds_pixart_alpha.pth')

        print(f'finish embedding')
        with torch.no_grad():

            if args.sampling_algo == 'dpm-solver':
                # Create sampling noise:
                n = len(prompts)
                z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device)
                model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
                if args.version == "alpha":
                    dpm_solver = DPMS_alpha(model.forward_with_dpmsolver,
                                condition=caption_embs,
                                uncondition=null_y,
                                cfg_scale=cfg_scale,
                                model_kwargs=model_kwargs)
                elif args.version == "sigma":
                    dpm_solver = DPMS_sigma(model.forward_with_dpmsolver,
                                    condition=caption_embs,
                                    uncondition=null_y,
                                    cfg_scale=cfg_scale,
                                    model_kwargs=model_kwargs)
                samples = dpm_solver.sample(
                    z,
                    steps=sample_steps,
                    order=2,
                    skip_type="time_uniform",
                    method="multistep",
                )
            else:
                raise NotImplementedError
            
        # ptq_config_file = os.path.join(args.ptq_config_path, f"{args.version}/pixart-dpm_{args.exp_name}.yml")
        ptq_config_file = args.ptq_config
        logger = logging.getLogger(__name__)
        config = OmegaConf.load(ptq_config_file)
        assert(config.conditional)

        # ======================================================
        # Build quantized model
        # ======================================================

        wq_params = config.quant.weight.quantizer
        aq_params = config.quant.activation.quantizer

        if config.get('mixed_precision', False):
            wq_params['mixed_precision'] = config.mixed_precision
            aq_params['mixed_precision'] = config.mixed_precision

        if config.get('timestep_wise', False):
            aq_params['timestep_wise'] = config.timestep_wise
            assert config.calib_data.n_steps == sample_steps, "When PTQ, the scheduler steps should equal to timesteps to record"

        qnn = QuantModel(
            model=model, \
            weight_quant_params=wq_params,\
            act_quant_params=aq_params,\
            model_type="pixart"
        )
        qnn.cuda()
        qnn.eval()
        print(qnn)
        logger.info(qnn)

        if not config.quant.grad_checkpoint:
            logger.info('Not use gradient checkpointing for transformer blocks')
            qnn.set_grad_ckpt(False)

        logger.info(f"Sampling data from {config.calib_data.n_steps} timesteps for calibration")
        if args.calib_data_path is not None:
            config.calib_data.path = args.calib_data_path
        calib_data_ckpt = torch.load(os.path.join(config.calib_data.path,f"{args.version}_calib_data_{args.sampling_algo}.pt"), map_location='cpu')
        calib_data = get_quant_calib_data(config, calib_data_ckpt, config.calib_data.n_steps, model_type=config.model.model_type)
        del(calib_data_ckpt)
        gc.collect()

        # ======================================================
        # Prepare data for init the model
        # ======================================================

        calib_added_kwargs = {}
        if config.model.model_type == "pixart":
            calib_batch_size = config.calib_data.batch_size  # DEBUG: actually for weight quant, only bs=1 is enough
            logger.info(f"Calibration data shape: {calib_data[0].shape} {calib_data[1].shape} {calib_data[2].shape}")
            calib_xs, calib_ts, calib_cs, calib_masks = calib_data
            calib_added_kwargs["mask"] = calib_masks
        else:
            raise NotImplementedError

        # for smooth quant
        if aq_params.smooth_quant.enable:
            args.smoothq = aq_params.smooth_quant.enable
            qnn.set_smooth_quant(smooth_quant=False, smooth_quant_running_stat=False)
            smooth_quant_layer_list= ["blocks.27.mlp.fc2"] 
            qnn.set_layer_smooth_quant(model=qnn, module_name_list=smooth_quant_layer_list, smooth_quant=True, smooth_quant_running_stat=True)

        # ======================================================
        # 6. get the quant params (training-free), using the calibration data
        # ======================================================
        # with torch.no_grad():
        # --- get temp kwargs -----
        if config.model.model_type == "pixart" and args.sampling_algo == "iddpm" or args.sampling_algo == "dpm-solver":
            dict(y=torch.cat([caption_embs, null_y]),
                                cfg_scale=cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
            tmp_kwargs = dict(cfg_scale=cfg_scale,
                                data_info={'img_hw': hw, 'aspect_ratio': ar},
                                mask=calib_added_kwargs["mask"][:calib_batch_size].cuda())
        else:
            tmp_kwargs = calib_added_kwargs
            
        _ = qnn(calib_xs[:calib_batch_size].cuda(), calib_ts[:calib_batch_size].cuda(), calib_cs[:calib_batch_size].cuda(), **tmp_kwargs)
        qnn.set_module_name_for_quantizer(module=qnn.model)  # add the module name as attribute for each quantizer

        # --- the weight quantization -----
        qnn.set_quant_state(True, False) # enable weight quantization, disable act quantization
        _ = qnn(calib_xs[:calib_batch_size].cuda(), calib_ts[:calib_batch_size].cuda(), calib_cs[:calib_batch_size].cuda(), **tmp_kwargs)
        logger.info("weight initialization done!")
        qnn.set_quant_init_done('weight')
        torch.cuda.empty_cache()

        # --- the activation quantization -----
        # by default, use the running_mean of calibration data to determine activation quant params
        qnn.set_quant_state(True, True) # quantize activation with fixed quantized weight
        fp_layer_list = ['x_embedder', 't_embedder', 't_block', 'y_embedder', 'csize_embedder', 'ar_embedder']
        qnn.fp_layer_list = fp_layer_list
        qnn.set_layer_quant(model=qnn, module_name_list=fp_layer_list, quant_level='per_layer', weight_quant=False, act_quant=False, prefix="")
    
        logger.info('Running stat for activation quantization')
        if aq_params.get('dynamic',False):
                logger.info('Adopting dynamic quant params, skip calculating fixed quant params')
        else:
            if not config.get('timestep_wise', False):
                # Normal activation calibration, walk through all calib data
                inds = np.arange(calib_xs.shape[0])
                tmp_kwargs = dict(fcf=cfg_scale,
                                data_info={'img_hw': hw, 'aspect_ratio': ar},)
                rounds = int(calib_xs.size(0) / calib_batch_size)

                for i in trange(rounds):
                    if config.model.model_type == "pixart":
                        _ = qnn(calib_xs[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),
                            calib_ts[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),
                            calib_cs[inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),
                            mask=calib_masks[inds[i * calib_batch_size:(i + 1) * calib_batch_size]][::2].cuda(),  # INFO: original opensora model takes in 2*bs conds(y) and bs mask
                            **tmp_kwargs
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
                tmp_kwargs = dict(fcf=cfg_scale,
                                data_info={'img_hw': hw, 'aspect_ratio': ar},)
                for i_ts in trange(calib_n_steps):
                    assert torch.all(calib_ts[i_ts,:] == calib_ts[i_ts,0])  # ts have the same timestepe_id
                    qnn.set_timestep_for_quantizer(calib_ts[i_ts,0].item())
                    for i in range(rounds):
                        if config.model.model_type == "opensora" or config.model.model_type == "pixart":
                            _ = qnn(\
                                    calib_xs[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                                    calib_ts[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size]].cuda(),\
                                    calib_cs[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                                    mask=calib_masks[i_ts, inds[i * calib_batch_size:(i + 1) * calib_batch_size],:].cuda(),\
                                    **tmp_kwargs
                            )
                        else:
                            raise NotImplementedError

                # INFO: re-fill the 1000 timesteps quant params if timestep-wise 
                if config.get('timestep_wise', False):
                    qnn.repeat_timestep_wise_quant_params(calib_ts)
        qnn.set_quant_init_done('activation')
        logger.info("activation initialization done!")
        torch.cuda.empty_cache()


        # ----------------------- get the quant params (training opt), using the calibration data -------------------------------------
        weight_optimization = False
        if config.quant.weight.optimization is not None:
            if config.quant.weight.optimization.params is not None:
                weight_optimization = True
        act_optimization = False
        if config.quant.activation.optimization is not None:
            if config.quant.activation.optimization.params is not None:
                act_optimization = True
        use_optimization = any([weight_optimization, act_optimization])

        if not use_optimization:  # no need for optimization-based quantization
            pass
        else:
            # INFO: get the quant parameters
            qnn.train()  # setup the train_mode
            opt_d = {}
            if weight_optimization:
                opt_d['weight'] = getattr(config.quant,'weight').optimization.params.keys()
            else:
                opt_d['weight'] = None
            if act_optimization:
                opt_d['activation'] = getattr(config.quant,'activation').optimization.params.keys()
            else:
                opt_d['activation'] = None
            qnn.replace_quant_buffer_with_parameter(opt_d)

            if config.quant.weight.optimization.joint_weight_act_opt:  # INFO: optimize all quant params together
                assert weight_optimization and act_optimization
                qnn.set_quant_state(True, True)
                opt_target = 'weight_and_activation'
                param_types = {
                        'weight': list(config.quant.weight.optimization.params.keys()),
                        'activation': list(config.quant.activation.optimization.params.keys())
                        }
                if 'alpha' in param_types['weight']:
                    assert config.quant.weight.quantizer.round_mode == 'learned_hard_sigmoid'  # check adaround stat
                if 'alpha' in param_types['activation']:
                    assert config.quant.activation.quantizer.round_mode == 'learned_hard_sigmoid'  # check adaround stat
                model_reconstruction(qnn,qnn,calib_data,config,param_types,opt_target)
                logger.info("Finished optimizing param {} for layer's {}, saving temporary checkpoint...".format(param_types,opt_target))
                torch.save(qnn.get_quant_params_dict(), os.path.join(save_root, "ckpt.pth"))

            else:  # INFO: sequantially quantize weight and activation quant params

                # --- the weight quantization (with optimization) -----
                if not weight_optimization:
                    logger.info("No quant parmas, skip optimizing weight quant parameters")
                else:
                    qnn.set_quant_state(True, False)  # use FP activation
                    opt_target = 'weight'
                    # --- unpack the config ----
                    param_types = list(config.quant.weight.optimization.params.keys())
                    if 'alpha' in param_types:
                        assert config.quant.weight.quantizer.round_mode == 'learned_hard_sigmoid'  # check adaround stat
                    # INFO: recursive iter through all quantizers, for weight/act quantizer, optimize the delta & alpha (if any) together
                    model_reconstruction(qnn,qnn,calib_data,config,param_types,opt_target)  # DEBUG_ONLY
                    logger.info("Finished optimizing param {} for layer's {}, saving temporary checkpoint...".format(param_types, opt_target))
                    torch.save(qnn.get_quant_params_dict(), os.path.join(save_root, "ckpt.pth"))

                # --- the activation quantization (with optimization) -----
                if not act_optimization:
                    logger.info("No quant parmas, skip optimizing activation quant parameters")
                else:
                    qnn.set_quant_state(True, True)  # use FP activation
                    opt_target = 'activation'
                    # --- unpack the config ----
                    param_types = list(config.quant.activation.optimization.params.keys())
                    if 'alpha' in param_types:
                        assert config.quant.weight.quantizer.round_mode == 'learned_hard_sigmoid'  # check adaround stat
                    # INFO: recursive iter through all quantizers, for weight/act quantizer, optimize the delta & alpha (if any) together
                    model_reconstruction(qnn,qnn,calib_data,config,param_types,opt_target)  # DEBUG_ONLY
                    logger.info("Finished optimizing param {} for layer's {}, saving temporary checkpoint...".format(param_types, opt_target))
                    torch.save(qnn.get_quant_params_dict(), os.path.join(save_root, "ckpt.pth"))

            qnn.replace_quant_parameter_with_buffers(opt_d)  # replace back to buffer for saving

        # save the quant params
        logger.info("Saving calibrated quantized PixArt model")
        quant_params_dict = qnn.get_quant_params_dict()
        torch.save(quant_params_dict, os.path.join(save_root, "ckpt.pth"))


if __name__ == '__main__':
    args = get_args()
    # Setup PyTorch:
    seed = args.seed
    set_env(seed)
    # torch.cuda.set_device(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    # only support fixed latent size currently
    latent_size = args.image_size // 8
    max_sequence_length = {"alpha": 120, "sigma": 300}[args.version]
    pe_interpolation = {256: 0.5, 512: 1, 1024: 2}     # trick for positional embedding interpolation
    micro_condition = True if args.version == 'alpha' and args.image_size == 1024 else False
    sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    weight_dtype = torch.float32  # when needing optimization. use FP32 for calib
    print(f"Inference with {weight_dtype}")

    # model setting
    micro_condition = True if args.version == 'alpha' and args.image_size == 1024 else False
    if args.image_size in [512, 1024, 2048, 2880]:
        model = PixArtMS_XL_2(
            input_size=latent_size,
            pe_interpolation=pe_interpolation[args.image_size],
            micro_condition=micro_condition,
            model_max_length=max_sequence_length,
        ).to(device)
    else:
        model = PixArt_XL_2(
            input_size=latent_size,
            pe_interpolation=pe_interpolation[args.image_size],
            model_max_length=max_sequence_length,
        ).to(device)
    print(model)
    print("Generating sample from ckpt: %s" % args.model_path)
    state_dict = find_model(args.model_path)
    if 'pos_embed' in state_dict['state_dict']:
        del state_dict['state_dict']['pos_embed']
    missing, unexpected = model.load_state_dict(state_dict['state_dict'], strict=False)
    print('Missing keys: ', missing)
    print('Unexpected keys', unexpected)
    model.eval()
    model.to(weight_dtype)
    base_ratios = eval(f'ASPECT_RATIO_{args.image_size}_TEST')

    if args.version == 'sigma':
        if args.sdvae:
            # pixart-alpha vae link: https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/sd-vae-ft-ema
            vae = AutoencoderKL.from_pretrained("output/pretrained_models/sd-vae-ft-ema").to(device).to(weight_dtype)
        else:
            # pixart-Sigma vae link: https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers/tree/main/vae
            vae = AutoencoderKL.from_pretrained(f"{args.pipeline_load_from}/vae").to(device).to(weight_dtype)

        tokenizer = T5Tokenizer.from_pretrained(args.pipeline_load_from, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(args.pipeline_load_from, subfolder="text_encoder").to(device)
        null_caption_token = tokenizer("", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
        null_caption_embs = text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]
    elif args.version == 'alpha':
        vae = AutoencoderKL.from_pretrained(os.path.join(args.pipeline_load_from, "sd-vae-ft-ema")).to(device).to(weight_dtype)
        if args.precomputed_text_embeds is not None:
            pass
        else:   
            t5 = T5Embedder(device=device, local_cache=True, cache_dir=os.path.join(args.pipeline_load_from, "t5-v1_1-xxl"), torch_dtype=torch.float)


    work_dir = "."
    
    # data setting
    with open(args.txt_file, 'r') as f:
        items = [item.strip() for item in f.readlines()]

    # img save setting
    try:
        epoch_name = re.search(r'.*epoch_(\d+).*', args.model_path).group(1)
        step_name = re.search(r'.*step_(\d+).*', args.model_path).group(1)
    except:
        epoch_name = 'unknown'
        step_name = 'unknown'

    # smooth = "_s" if args.smoothq else ""
    # save_root = f"./quant_models/{args.version}/{args.ptq_config}{smooth}"
    save_root = os.path.join(args.save_path,f"{args.version}/{args.exp_name}")
    os.makedirs(save_root, exist_ok=True)

    outpath = save_root
    if not os.path.exists(outpath):
        os.makedirs(outpath, exist_ok=True)
    print(outpath)

    # INFO: add backup file and backup cfg into logpath for debug
    if os.path.exists(os.path.join(outpath,'config.yaml')):
        os.remove(os.path.join(outpath,'config.yaml'))
    # ptq_config_file = os.path.join(args.ptq_config_path, f"{args.version}/pixart-dpm_{args.exp_name}.yml")
    ptq_config_file = args.ptq_config
    shutil.copy(ptq_config_file, os.path.join(outpath,'config.yaml'))
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

    visualize(items, args.bs, sample_steps, args.cfg_scale)
