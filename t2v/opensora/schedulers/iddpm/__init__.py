from functools import partial

import torch

from opensora.registry import SCHEDULERS

from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps


@SCHEDULERS.register_module("iddpm")
class IDDPM(SpacedDiffusion):
    def __init__(
        self,
        num_sampling_steps=None,
        timestep_respacing=None,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=False,
        predict_xstart=False,
        learn_sigma=True,
        rescale_learned_sigmas=False,
        diffusion_steps=1000,
        cfg_scale=4.0,
    ):
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
        if num_sampling_steps is not None:
            assert timestep_respacing is None
            timestep_respacing = str(num_sampling_steps)
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [diffusion_steps]
        super().__init__(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
            model_var_type=(
                (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            # rescale_timesteps=rescale_timesteps,
        )

        self.cfg_scale = cfg_scale

    def sample(
        self,
        model,
        text_encoder,
        sampler_type,
        z_size,
        prompts,
        device,
        return_trajectory=False,
        additional_args=None,
        init_noise=None
    ):
        n = len(prompts)
        z = torch.randn(n, *z_size, device=device) if init_noise is None else init_noise.to(device)
        z = torch.cat([z, z], 0)

        # INFO: support loading precomputed text_embeds
        if additional_args is not None:
            if "precompute_text_embeds" in additional_args.keys():
                choose_idx = additional_args['batch_ids']
                model_args = additional_args['precompute_text_embeds'].copy()
                text_embeds_shape = model_args['y'].shape
                # handling of drop_last
                if choose_idx.max() > text_embeds_shape[0]:
                    model_args['y'] = model_args['y'][choose_idx[0]:,:].permute([1,0,2,3,4]).reshape([-1,\
                        text_embeds_shape[2],text_embeds_shape[3],text_embeds_shape[4]])
                    model_args['mask'] = model_args['mask'][choose_idx[0]:,:]
                else:
                    model_args['y'] = model_args['y'][choose_idx,:].permute([1,0,2,3,4]).reshape([len(choose_idx)*text_embeds_shape[1],\
                        text_embeds_shape[2],text_embeds_shape[3],text_embeds_shape[4]])
                    model_args['mask'] = model_args['mask'][choose_idx,:]
            else:
                model_args = text_encoder.encode(prompts)
                y_null = text_encoder.null(n)
                model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        else:
            model_args = text_encoder.encode(prompts)
            y_null = text_encoder.null(n)
            model_args["y"] = torch.cat([model_args["y"], y_null], 0)

        # merge additional args into model_args
        if additional_args is not None:
            # INFO: with precomputed text_embeds, leave out the batch_ids and precompute_text_embeds
            additional_args_ = additional_args.copy()
            for k_ in ['precompute_text_embeds','batch_ids']:
                if k_ in additional_args_.keys():
                    del additional_args_[k_]
            model_args.update(additional_args_)
        forward = partial(forward_with_cfg, model, cfg_scale=self.cfg_scale, return_trajectory=return_trajectory)
        if sampler_type == "ddim":
            samples = self.ddim_sample_loop(
                forward,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_args,
                progress=True,
                return_trajectory=return_trajectory,
                device=device,
            )
        elif sampler_type == "iddpm":
            samples = self.p_sample_loop(
                forward,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_args,
                progress=True,
                return_trajectory=return_trajectory,
                device=device,
            )
        else:
            raise NotImplementedError(f"{sampler_type} is not a supported sampler type!")
        if not return_trajectory:
            samples, _ = samples.chunk(2, dim=0)
            return samples
        else:
            samples, calib_data, out_data = samples
            samples, _ = samples.chunk(2, dim=0)
            return samples, calib_data, out_data


def forward_with_cfg(model, x, timestep, y, cfg_scale, return_trajectory=False, **kwargs):
    # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb

    # INFO: split the model forward along the batch
    # to enable different quant params for cond & uncond branch
    CFG_SPLIT = model.cfg_split if hasattr(model, 'cfg_split') else False  # DIRTY: read the cfg_split cfg from model
    if CFG_SPLIT:
        # DEBUG_ONLY
        half = x[: len(x) // 2] # actually use the 1st half of x
        combined = torch.cat([half, half], dim=0)
        # model_out_gt = model.forward(combined, timestep, y, **kwargs) # model forward

        # SPLIT the y
        y_shape  = y.shape
        y = y.reshape([2,y_shape[0]//2]+list(y_shape[1:]))
        timestep = timestep.reshape([2,-1])
        y_cond, y_uncond = y.unbind(0)
        t_cond, t_uncond = timestep.unbind(0)

        half = x[: len(x) // 2] # use the 1st half of x, same for cond and uncond

        model_output_cond = model.forward(half, t_cond, y_cond, **kwargs)
        model_output_uncond = model.forward(half, t_uncond, y_uncond, **kwargs)

        model_out = torch.cat([model_output_cond, model_output_uncond], dim=0)
    else:
        half = x[: len(x) // 2] # actually use the 1st half of x
        combined = torch.cat([half, half], dim=0)
        model_out = model.forward(combined, timestep, y, **kwargs) # model forward


    # INFO: for PTQD, the correlated noise correction & the 

    ks = torch.load('./t2v/rebuttal_files/k_for_each_timestep.pth')
    # calib_quant_noise = torch.load('./rebuttal_files/calibrated_quant_noise.pth')
    # the correlated noise correction
    timestep_idx = (999 - timestep[0]) // 50  # for 20 timesteps
    model_out = model_out / (1+ks[timestep_idx])
    # the bias correction
    # model_out = model_out - calib_quant_noise[timestep]

    save_model_out = model_out
    model_out = model_out["x"] if isinstance(model_out, dict) else model_out
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    if return_trajectory:
        return torch.cat([eps, rest], dim=1), save_model_out
    return torch.cat([eps, rest], dim=1)
