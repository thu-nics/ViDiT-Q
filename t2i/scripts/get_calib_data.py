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
from tqdm import tqdm
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from transformers import T5EncoderModel, T5Tokenizer

from diffusion.model.utils import prepare_prompt_ar
from diffusion import IDDPM, DPMS_alpha, DPMS_sigma, SASolverSampler
from tools.download import find_model
from diffusion.model.t5 import T5Embedder
from diffusion.model.nets import PixArtMS_XL_2, PixArt_XL_2
from diffusion.data.datasets import get_chunks
from diffusion.data.datasets.utils import *



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument('--version', default='sigma', type=str)
    parser.add_argument(
        "--pipeline_load_from", default='output/pretrained_models/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument('--txt_file', default='t2i/asset/calib.txt', type=str)
    parser.add_argument('--model_path', default='t2i/output/pretrained_models/PixArt-XL-2-1024x1024.pth', type=str)
    parser.add_argument('--sdvae', action='store_true', help='sd vae')
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--sampling_algo', default='dpm-solver', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default='custom', type=str)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--save_name', default='test_sample', type=str)
    parser.add_argument('--save_path', required=False, type=str)
    parser.add_argument('--save_text_embeds_only', action='store_true', help='save the text embeds instead of the calib data')
    parser.add_argument('--precomputed_text_embeds', default=None, type=str)
    # parser.add_argument('--gpu', default=0, type=int)

    return parser.parse_args()

def get_idx_from_prompt_list(subset_list, larger_list):
    index_dict = {value: index for index, value in enumerate(larger_list)}
    indexes = [index_dict.get(item, -1) for item in subset_list]
    return indexes

def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, args.image_size, args.image_size)

@torch.inference_mode()
def visualize(items, bs, sample_steps, cfg_scale, args):
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)

    # optionally save the text embeds
    if args.save_text_embeds_only is not None:
        save_d = {}
        save_d['prompts'] = items
        for name_ in ['caption_embs','emb_masks','null_y']:
            save_d[name_] = []

    if args.precomputed_text_embeds is not None:
        save_d = torch.load(args.precomputed_text_embeds)

    calib_data = {}
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
            if args.precomputed_text_embeds is not None:
                indexes = get_idx_from_prompt_list(prompts, save_d['prompts'])
                caption_embs = save_d['caption_embs'][indexes,:]
                emb_masks = save_d['emb_masks'][indexes,:]
                null_y = save_d['null_y'][indexes,:]
            else:
                caption_token = tokenizer(prompts, max_length=max_sequence_length, padding="max_length", truncation=True,
                                        return_tensors="pt").to(device)
                caption_embs = text_encoder(caption_token.input_ids, attention_mask=caption_token.attention_mask)[0]
                caption_embs = caption_embs[:, None]
                emb_masks = caption_token.attention_mask
                null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]

                if args.save_text_embeds_only:
                    # INFO: optional save the text_embeds
                    save_d['caption_embs'].append(caption_embs)  # [bs, text_embed_size]
                    save_d['emb_masks'].append(emb_masks)
                    save_d['null_y'].append(null_y)

        elif args.version == 'alpha':
            if args.precomputed_text_embeds is not None:
                indexes = get_idx_from_prompt_list(prompts, save_d['prompts'])
                caption_embs = save_d['caption_embs'][indexes,:]
                emb_masks = save_d['emb_masks'][indexes,:]
                null_y = save_d['null_y'][indexes,:]
            else:
                caption_embs, emb_masks = t5.get_text_embeddings(prompts)
                caption_embs = caption_embs.float()[:,None]
                null_y = model.y_embedder.y_embedding[None].repeat(len(prompts), 1, 1)[:, None]

                if args.save_text_embeds_only:
                    # INFO: optional save the text_embeds
                    save_d['caption_embs'].append(caption_embs)  # [bs, text_embed_size]
                    save_d['emb_masks'].append(emb_masks)
                    save_d['null_y'].append(null_y)

        print(f'finish embedding')

        if args.save_text_embeds_only:  # skip the solver
            pass
        else:
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
                    samples, _, cur_calib_data = dpm_solver.sample(
                        z,
                        steps=sample_steps,
                        order=2,
                        skip_type="time_uniform",
                        method="multistep",
                        return_intermediate=True,
                    )
                    cur_calib_data["ts"] = torch.tensor(cur_calib_data["ts"][:-1]).reshape((sample_steps,1))
                    cur_calib_data["xs"] = torch.cat(cur_calib_data["xs"][:-1], 0)
                    cur_calib_data["xs"] = cur_calib_data["xs"].reshape((sample_steps, bs, 4, latent_size_h, latent_size_w))
                    cur_calib_data["cond_emb"] = caption_embs.repeat(sample_steps, 1, 1, 1).reshape((sample_steps, bs, 1, max_sequence_length, 4096))
                    cur_calib_data["mask"] = emb_masks.repeat(sample_steps, 1).reshape((sample_steps, bs, max_sequence_length))
                    for key in cur_calib_data:
                        if not key in calib_data.keys():
                            calib_data[key] = cur_calib_data[key]
                        else:
                            calib_data[key] = torch.cat([cur_calib_data[key], calib_data[key]], dim=1)    
                        # print(key, calib_data[key].shape)
                else:
                    raise NotImplementedError

            samples = samples.to(weight_dtype)
            samples = vae.decode(samples / vae.config.scaling_factor).sample
            torch.cuda.empty_cache()
            # Save images:
            os.umask(0o000)  # file permission: 666; dir permission: 777

            # for i, sample in enumerate(samples):
            #     save_path = os.path.join(save_root, f"{prompts[i][:100]}.jpg")
            #     print("Saving path: ", save_path)
            #     try:
            #         save_image(sample, save_path, nrow=1, normalize=True, value_range=(-1, 1))
            #     except:
            #         continue

    # INFO: optional save text embeds
    if args.save_text_embeds_only:
        for name_ in ['caption_embs','emb_masks','null_y']:
            save_d[name_] = torch.cat(save_d[name_], dim=0)
        torch.save(save_d, os.path.join('./t2i/asset', f"text_embeds_{args.version}_{args.txt_file.split('/')[-1].strip('.txt')}.pth"))

    # when saving the text embeds, skip storing the large calib data
    # since the text embeds are not necessarily used for calibration, could be used for quant_infer also.
    if not args.save_text_embeds_only:
        torch.save(calib_data, os.path.join(save_path, f"{args.version}_calib_data_{args.sampling_algo}.pt"))


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
    pe_interpolation = {256: 0.5, 512: 1, 1024: 2, 2048: 4}     # trick for positional embedding interpolation
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

        if args.precomputed_text_embeds is not None:
            pass
        else:
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

    # save_root=f"{work_dir}/quant_models/calib_data/"
    # os.makedirs(save_root, exist_ok=True)
    visualize(items, args.bs, sample_steps, args.cfg_scale, args)
