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
from qdiff.utils import load_quant_params

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
    parser.add_argument('--save_path', default='./', type=str)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--quant_weight', action='store_true', help="if to quantize the weight")
    parser.add_argument('--quant_act', action='store_true', help="if to quantize the act")
    parser.add_argument('--quant_path', default='./quant_models', type=str)
    parser.add_argument('--bitwidth_setting', default='', type=str)
    parser.add_argument('--caption_emb_path', default='./t2i/asset/samples2.pt', type=str)

    return parser.parse_args()

def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)

@torch.inference_mode()
def visualize(items, bs, sample_steps, cfg_scale):

    for chunk in tqdm(list(get_chunks(items, bs)), unit='batch'):

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
            caption_embs, emb_masks = t5.get_text_embeddings(prompts)
            caption_embs = caption_embs.float()[:,None]
            null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]

        print(f'finish embedding')

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
        elif args.sampling_algo == 'iddpm':
            # Create sampling noise:
            n = len(prompts)
            z = torch.randn(n, 4, latent_size_h, latent_size_w, device=device).repeat(2, 1, 1, 1)
            model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                cfg_scale=cfg_scale, data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
            diffusion = IDDPM(str(sample_steps))
            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        elif args.sampling_algo == 'sa-solver':
            # Create sampling noise:
            n = len(prompts)
            model_kwargs = dict(data_info={'img_hw': hw, 'aspect_ratio': ar}, mask=emb_masks)
            sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
            samples = sa_solver.sample(
                S=25,
                batch_size=n,
                shape=(4, latent_size_h, latent_size_w),
                eta=1,
                conditioning=caption_embs,
                unconditional_conditioning=null_y,
                unconditional_guidance_scale=cfg_scale,
                model_kwargs=model_kwargs,
            )[0]

        samples = samples.to(weight_dtype)
        samples = vae.decode(samples / vae.config.scaling_factor).sample
        torch.cuda.empty_cache()
        # Save images:
        os.umask(0o000)  # file permission: 666; dir permission: 777
        for i, sample in enumerate(samples):
            save_path = os.path.join(save_root, f"{prompts[i][:100]}.jpg")
            print("Saving path: ", save_path)
            try:
                save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))
            # plot_distributions(qnn, os.path.join(save_root, f"dist-{alpha}-"), True)
            except:
                continue

if __name__ == '__main__':
    args = get_args()
    # Setup PyTorch:
    seed = args.seed
    set_env(seed)
    torch.cuda.set_device(args.gpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    # only support fixed latent size currently
    latent_size = args.image_size // 8
    max_sequence_length = {"alpha": 120, "sigma": 300}[args.version]
    pe_interpolation = {256: 0.5, 512: 1, 1024: 2}     # trick for positional embedding interpolation
    micro_condition = True if args.version == 'alpha' and args.image_size == 1024 else False
    sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    weight_dtype = torch.float16
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

    use_quant_act = args.quant_act
    use_quant_weight = args.quant_weight
    # Update ptq_config for full precision weight/act
    # quant_path = os.path.join(args.quant_ckpt_path, f"{args.version}/{args.ptq_config}")
    quant_path = args.quant_path
    print(quant_path)
    if not os.path.exists(quant_path):
        print(f"Error: {quant_path} does not exist")
        sys.exit(1)
    if not use_quant_act:
        bitwidth_setting = args.bitwidth_setting[:2] + "a16"
    elif not use_quant_weight:
        bitwidth_setting = "w16" + args.bitwidth_setting[-2:]
    else: 
        bitwidth_setting = args.bitwidth_setting

    logger = logging.getLogger(__name__)
    ptq_config_file = os.path.join(quant_path, f"config.yaml")
    config = OmegaConf.load(ptq_config_file)
    assert(config.conditional)

    wq_params = config.quant.weight.quantizer
    aq_params = config.quant.activation.quantizer

    if config.get('mixed_precision', False):
        wq_params['mixed_precision'] = config.mixed_precision
        aq_params['mixed_precision'] = config.mixed_precision

    qnn = QuantModel(
        model=model, \
        weight_quant_params=wq_params,\
        act_quant_params=aq_params,\
        model_type = "pixart"
    )
    qnn.cuda()
    qnn.eval()
    logger.info(qnn)

    qnn.set_quant_state(False, False)
    hw = torch.tensor([[args.image_size, args.image_size]], dtype=torch.float, device=device)
    ar = torch.tensor([[1.]], device=device)
    # for smooth quant
    if aq_params.smooth_quant.enable:
        qnn.set_smooth_quant(smooth_quant=False, smooth_quant_running_stat=False)
    with torch.no_grad():
        if "pixart" in config.model.model_id:
            _ = qnn(torch.randn((1, 4, 128, 128)).to("cuda"), torch.randint(0, 1000, (1,)).to("cuda"), torch.randn((1, 1, 120, 4096)).to("cuda"), mask=torch.ones((1, 120)).to("cuda"), data_info={'img_hw': hw, 'aspect_ratio': ar}, max_seqlen_k=120)
        else:
            raise NotImplementedError
        qnn.set_quant_state(use_quant_weight, use_quant_act)
        fp_layer_list = ['x_embedder', 't_embedder', 't_block', 'y_embedder', 'csize_embedder', 'ar_embedder']
        qnn.set_layer_quant(model=qnn, module_name_list=fp_layer_list, quant_level='per_layer', weight_quant=False, act_quant=False, prefix="")
        
        if aq_params.smooth_quant.enable:
            qnn.set_smooth_quant(smooth_quant=False, smooth_quant_running_stat=False)
            smooth_quant_layer_list= ["blocks.27.mlp.fc2"] 
            qnn.set_layer_smooth_quant(model=qnn, module_name_list=smooth_quant_layer_list, smooth_quant=True, smooth_quant_running_stat=True)

        qnn.set_quant_init_done('weight')
        qnn.set_quant_init_done('activation')

        load_quant_params(qnn, os.path.join(quant_path, "ckpt.pth"))
        qnn.cuda()
        qnn.to(weight_dtype)

    save_root = os.path.join(args.save_path, f"{args.version}/{bitwidth_setting}/generated_imgs")
    # save_root = args.save_path
    os.makedirs(save_root, exist_ok=True)

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
    outpath = save_root
    
    # INFO: add backup file and backup cfg into logpath for debug
    if os.path.exists(os.path.join(outpath,'config.yaml')):
        os.remove(os.path.join(outpath,'config.yaml'))
    # ptq_config_file = os.path.join(quant_path, f"config.yaml")
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