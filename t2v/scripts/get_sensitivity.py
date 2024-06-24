import os
import sys
# sys.path.append("/home/fangtongcheng/quant_dit/diffuser-dev")
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
from qdiff.models.quant_layer import QuantLayer
from qdiff.utils import load_quant_params
import pytorch_lightning as pl

import inspect
import os

logger = logging.getLogger(__name__)


def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


def layer_set_quant(
                model,
                prompts,
                cfg,
                scheduler,
                quantized_model, 
                text_encoder,
                vae,
                save_dir,
                dtype,
                model_args,
                latent_size,
                pre_textemb,
                device,
                weight_quant=True, 
                act_quant=False, 
                cur_bit=0, 
                prefix="",
                ):
    '''
    compute the error of the output of the model with a certain layer quantized
    '''

    base_dir = os.path.join(save_dir, f"layers")
    os.makedirs(save_dir, exist_ok=True)
    # with open(os.path.join(save_dir, "layer_names.txt"), "a") as f:
    for name, module in model.named_children():
        full_name = prefix + name if prefix else name
        layer_ignore = ['embedder', 'final', 't_block']  # keep those layers fp
        # logger.info(f"{name} {)}")
        if isinstance(module, QuantLayer):
            if all(element not in full_name for element in layer_ignore):
            # if not 'ff' in full_name and not 'attn2' in full_name:
                module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                torch.cuda.empty_cache()
                logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
                # f.write(full_name + "\n")
                # mse_mean = 0
                # sqnr_mean = 0
                # for idx, input_data in enumerate(input_list):
                save_dir = os.path.join(base_dir, f"{full_name}")
                os.makedirs(save_dir, exist_ok=True)

                with torch.no_grad():
                    video_generate(
                                    prompts, 
                                    cfg,
                                    scheduler,
                                    quantized_model,
                                    text_encoder,
                                    vae,
                                    save_dir,
                                    dtype,
                                    model_args,
                                    latent_size,
                                    pre_textemb,
                                    device
                                    )

                quantized_model.set_quant_state(False, False)
        else:
            # layer_set_quant(model=module, quantized_model=quantized_model, weight_quant=weight_quant, act_quant=act_quant, cur_bit=cur_bit, prefix=full_name+".")
            layer_set_quant(
                            model=module,
                            prompts=prompts,
                            cfg=cfg,
                            scheduler=scheduler,
                            quantized_model=quantized_model, 
                            text_encoder=text_encoder,
                            vae=vae,
                            save_dir=save_dir,
                            dtype=dtype,
                            model_args=model_args,
                            latent_size=latent_size,
                            pre_textemb=pre_textemb,
                            device=device,
                            weight_quant=weight_quant, 
                            act_quant=act_quant, 
                            cur_bit=cur_bit, 
                            prefix=full_name+".",
                            )


def group_set_quant(
                qnn,
                prompts,
                cfg,
                opt,
                scheduler,
                text_encoder,
                vae,
                dtype,
                model_args,
                latent_size,
                pre_textemb,
                device,
                weight_quant=True, 
                act_quant=False,  
                ):
    quant_group_list = ['attn', 'cross_attn', 'mlp', 'attn_temp']
    unquant_group_list = ['embedder', 'final', 't_block']

    for group_name in quant_group_list:
        torch.manual_seed(cfg.seed)
        save_dir = os.path.join(opt.save_dir, f"group/{group_name}")
        os.makedirs(save_dir, exist_ok=True)

        qnn.set_quant_state(False, False) 
        group_name_list = [group_name]
        qnn.set_layer_quant(model=qnn, group_list=group_name_list, group_ignore=unquant_group_list, quant_level='per_group', weight_quant=weight_quant, act_quant=act_quant, prefix="")
        
        with torch.no_grad():
            video_generate(
                            prompts=prompts, 
                            cfg=cfg,
                            scheduler=scheduler,
                            qnn=qnn,
                            text_encoder=text_encoder,
                            vae=vae,
                            save_dir=save_dir,
                            dtype=dtype,
                            model_args=model_args,
                            latent_size=latent_size,
                            pre_textemb=pre_textemb,
                            device=device,
                            )
        # sample_idx = 0
        # for i in range(0, len(prompts), cfg.batch_size):
        #     batch_prompts = prompts[i : i + cfg.batch_size]
        #     if pre_textemb is not None:  # also feed in the idxs for saved text_embeds
        #         model_args['batch_ids'] = torch.arange(i,i+cfg.batch_size)
        #     samples = scheduler.sample(
        #         qnn,
        #         text_encoder,
        #         sampler_type=cfg.sampler,
        #         z_size=(vae.out_channels, *latent_size),
        #         prompts=batch_prompts,
        #         device=device,
        #         additional_args=model_args,
        #     )
        #     samples = vae.decode(samples.to(dtype))

        #     for idx, sample in enumerate(samples):
        #         print(f"Prompt: {batch_prompts[idx]}")
        #         save_path = os.path.join(save_dir, f"sample_{sample_idx}")
        #         save_sample(sample, fps=cfg.fps, save_path=save_path)
        #         sample_idx += 1


def timestep_set_quant(
            qnn,
            prompts,
            cfg,
            opt,
            scheduler,
            text_encoder,
            vae,
            dtype,
            model_args,
            latent_size,
            pre_textemb,
            device,
            ):
    split_timestep = 4  # split the timesteps into N parts
    n_step = cfg.scheduler.num_sampling_steps
    quant_time_list = [i * (n_step) // split_timestep for i in range(split_timestep)]
    for i in range(len(quant_time_list)):
        qnn.set_quant_state(False, False)

        qnn.quant_start_t = quant_time_list[split_timestep-i-1]+(n_step//split_timestep-1)
        qnn.quant_end_t = quant_time_list[split_timestep-i-1]

        save_dir = os.path.join(opt.save_dir, f"t_{n_step}_split_{split_timestep}/{qnn.quant_start_t}_{qnn.quant_end_t}")
        os.makedirs(save_dir, exist_ok=True)

        video_generate(
                        prompts, 
                        cfg, 
                        scheduler, 
                        qnn, 
                        text_encoder, 
                        vae, 
                        save_dir, 
                        dtype, 
                        model_args, 
                        latent_size, 
                        pre_textemb, 
                        device
                        )


def video_generate(prompts, cfg, scheduler, qnn, text_encoder, vae, save_dir, dtype, model_args, latent_size, pre_textemb, device):
    torch.manual_seed(cfg.seed)
    sample_idx = 0
    for i in range(0, len(prompts), cfg.batch_size):
        batch_prompts = prompts[i : i + cfg.batch_size]
        if pre_textemb is not None:  # also feed in the idxs for saved text_embeds
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

    log_path = os.path.join(opt.save_dir, "quant_inference_run.log")
    os.makedirs(opt.save_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
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
    pl.seed_everything(cfg.seed)
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
    use_weight_quant = False if wq_params is False else True
    # use_act_quant = False if aq_params is False else True
    use_weight_quant = not opt.skip_quant_weight
    use_act_quant = not opt.skip_quant_act

    if config.get('mixed_precision', False):
        # if use_weight_quant:
        wq_params['mixed_precision'] = config.mixed_precision
        # aq_params['mixed_precision'] = config.mixed_precision
        # print(aq_params)
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

    qnn.set_quant_state(False, False)
    # for smooth quant
    if aq_params.smooth_quant.enable:
        qnn.set_smooth_quant(smooth_quant=False, smooth_quant_running_stat=False)

    calib_added_cond = {} # It is not required for STDiT

    with torch.no_grad():
        if "opensora" in config.model.model_id:
            _ = qnn(torch.randn(1, 4, 16, 64, 64).cuda(), torch.randint(0, 1000, (1,)).cuda(), torch.randn(1, 1, 120, 4096).cuda(), mask=torch.ones(1, 120).cuda().to(torch.int64))
        else:
            raise NotImplementedError

    if opt.part_fp:
        fp_layer_list = ['embedder', 'final', 't_block']

    # for smooth quant
    if aq_params.smooth_quant.enable:
        qnn.set_smooth_quant(smooth_quant=False, smooth_quant_running_stat=False)
        smooth_quant_layer_list= ["blocks.27.mlp.fc2"] 
        qnn.set_layer_smooth_quant(model=qnn, module_name_list=smooth_quant_layer_list, smooth_quant=True, smooth_quant_running_stat=False)

    # set the init flag True, otherwise will recalculate params
    if opt.part_fp:
        qnn.set_quant_state(use_weight_quant, use_act_quant)
        qnn.use_weight_quant = use_weight_quant
        qnn.use_act_quant = use_act_quant
        qnn.set_layer_quant(model=qnn, module_name_list=fp_layer_list, quant_level='per_layer', weight_quant=False, act_quant=False, prefix="")
    else:
        qnn.set_quant_state(use_weight_quant, use_act_quant) # enable weight quantization, disable act quantization
        qnn.use_weight_quant = use_weight_quant
        qnn.use_act_quant = use_act_quant

    qnn.set_quant_init_done('weight')
    qnn.set_quant_init_done('activation')

    load_quant_params(qnn, opt.quant_ckpt)
    qnn.cuda()
    qnn.to(dtype)
    

    # ======================================================
    # 5. inference
    # ======================================================
    qnn.timestep_wise_quant = opt.timestep_wise_quant
    qnn.layer_wise_quant = opt.layer_wise_quant
    qnn.group_wise_quant = opt.group_wise_quant
    qnn.block_group_wise_quant = opt.block_group_wise_quant
    if PRECOMPUTE_TEXT_EMBEDS is not None:
        model_args['precompute_text_embeds'] = torch.load(cfg.precompute_text_embeds)

    # Quant a part of the model layers or timesteps
    if not opt.timestep_wise_quant:
        qnn.timestep_wise_quant = False
        if opt.group_wise_quant:
            qnn.group_wise_quant = True
            qnn.set_quant_state(False, False)
            group_set_quant(
                        qnn=qnn,
                        prompts=prompts,
                        cfg=cfg,
                        opt=opt,
                        scheduler=scheduler,
                        text_encoder=text_encoder,
                        vae=vae,
                        dtype=dtype,
                        model_args=model_args,
                        latent_size=latent_size,
                        pre_textemb=PRECOMPUTE_TEXT_EMBEDS,
                        device=device,
                        weight_quant=use_weight_quant, 
                        act_quant=use_act_quant,  
                    )

        elif opt.layer_wise_quant:
            qnn.layer_wise_quant = True
            qnn.set_quant_state(False, False)
            layer_set_quant(model=qnn,
                            prompts=prompts,
                            cfg=cfg,
                            scheduler=scheduler,
                            quantized_model=qnn,
                            text_encoder=text_encoder,
                            vae=vae,
                            save_dir=opt.save_dir,
                            dtype=dtype,
                            device=device,
                            model_args=model_args,
                            latent_size=latent_size,
                            pre_textemb=PRECOMPUTE_TEXT_EMBEDS,
                            weight_quant=use_weight_quant,
                            act_quant=use_act_quant,
                            # cur_bit=8
                            )
        
        elif opt.block_group_wise_quant:
            for bit_width in [4,6,8]:
                qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='weight')
                qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='act')

                qnn.block_group_wise_quant = True
                qnn.layer_wise_quant = False
                qnn.group_wise_quant = False

                # with open("./layer_names.txt", 'r', encoding='utf-8') as file:
                #     quant_layer_list = file.read().splitlines()

                split_timestep = 4  # Uniformly divide the time step into 'split_timestep' parts
                n_step = cfg.scheduler.num_sampling_steps
                quant_time_list = [i * (n_step) // split_timestep for i in range(split_timestep)]

                # 从txt文件中读取模型层名字
                with open('/home/fangtongcheng/quant_dit/diffuser-dev/layer_names.txt', 'r') as f:
                    layers = f.read().splitlines()

                group_name = opt.group_quant  ###############################################
                
                # for i in range(len(quant_time_list)):
                    # 遍历所有的模型层
                for j in range(28):
                    # 选择出名字中包含当前数字的层
                    selected_layers = [layer for layer in layers if f'model.blocks.{j}.' in layer]
                    # 进一步选择出名字中包含".attn."或"mlp"的层
                    selected_layers_group = [layer for layer in selected_layers if group_name in layer]

                    qnn.quant_layer_name = selected_layers_group

                    qnn.set_quant_state(False, False) 
                    qnn.set_layer_quant(model=qnn, module_name_list=qnn.quant_layer_name, quant_level='per_layer', weight_quant=qnn.use_weight_quant, act_quant=qnn.use_act_quant, prefix="")

                    save_dir = os.path.join(opt.save_dir, f"bit_{bit_width}/block_{j}")
                    os.makedirs(save_dir, exist_ok=True)

                    video_generate(
                                    prompts, 
                                    cfg, 
                                    scheduler, 
                                    qnn, 
                                    text_encoder, 
                                    vae, 
                                    save_dir, 
                                    dtype, 
                                    model_args, 
                                    latent_size, 
                                    PRECOMPUTE_TEXT_EMBEDS, 
                                    device
                                    )


        elif opt.block_wise_quant_progressively:
            # for bit_width in [4,6,8]:
            # qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='weight')
            # qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='act')

            # qnn.block_group_wise_quant = True
            # qnn.layer_wise_quant = False
            # qnn.group_wise_quant = False

            # with open("./layer_names.txt", 'r', encoding='utf-8') as file:
            #     quant_layer_list = file.read().splitlines()

            # split_timestep = 4  # Uniformly divide the time step into 'split_timestep' parts
            # n_step = cfg.scheduler.num_sampling_steps
            # quant_time_list = [i * (n_step) // split_timestep for i in range(split_timestep)]

            # 从txt文件中读取模型层名字
            with open('/home/fangtongcheng/quant_dit/diffuser-dev/layer_names.txt', 'r') as f:
                layers = f.read().splitlines()

            # group_name = opt.group_quant  ###############################################
            
            # for i in range(len(quant_time_list)):
            # 遍历所有的模型层
            qnn.set_quant_state(False, False)
            for j in range(28):
                # 选择出名字中包含当前数字的层
                selected_layers = [layer for layer in layers if f'model.blocks.{j}.' in layer]
                # 进一步选择出名字中包含".attn."或"mlp"的层
                # selected_layers_group = [layer for layer in selected_layers if group_name in layer]
                # 将这些层的名字添加到列表中
                # selected_layers_list.extend(selected_layers_group)

                # qnn.quant_start_t = quant_time_list[split_timestep-i-1]+(n_step//split_timestep-1)
                # qnn.quant_end_t = quant_time_list[split_timestep-i-1]

                # qnn.quant_layer_name = selected_layers_group
                qnn.set_layer_quant(model=qnn, module_name_list=selected_layers, quant_level='per_layer', weight_quant=use_weight_quant, act_quant=use_act_quant, prefix="")

                save_dir = os.path.join(opt.save_dir, f"blocks_progressively/block_0-block_{j}")
                os.makedirs(save_dir, exist_ok=True)
                # qnn.set_quant_state(False, False)

                video_generate(
                                prompts, 
                                cfg, 
                                scheduler, 
                                qnn, 
                                text_encoder, 
                                vae, 
                                save_dir, 
                                dtype, 
                                model_args, 
                                latent_size, 
                                PRECOMPUTE_TEXT_EMBEDS, 
                                device
                                )

        elif opt.block_wise_quant:

            # 从txt文件中读取模型层名字
            with open('/home/fangtongcheng/quant_dit/diffuser-dev/layer_names.txt', 'r') as f:
                layers = f.read().splitlines()

            for j in range(28):
                qnn.set_quant_state(False, False)
                # 选择出名字中包含当前数字的层
                selected_layers = [layer for layer in layers if f'model.blocks.{j}.' in layer]
                # 进一步选择出名字中包含".attn."或"mlp"的层
                # selected_layers_group = [layer for layer in selected_layers if group_name in layer]
                # 将这些层的名字添加到列表中
                # selected_layers_list.extend(selected_layers_group)

                # qnn.quant_start_t = quant_time_list[split_timestep-i-1]+(n_step//split_timestep-1)
                # qnn.quant_end_t = quant_time_list[split_timestep-i-1]

                # qnn.quant_layer_name = selected_layers_group
                qnn.set_layer_quant(model=qnn, module_name_list=selected_layers, quant_level='per_layer', weight_quant=use_weight_quant, act_quant=use_act_quant, prefix="")

                save_dir = os.path.join(opt.save_dir, f"blocks/block_{j}")
                os.makedirs(save_dir, exist_ok=True)
                # qnn.set_quant_state(False, False)

                video_generate(
                                prompts, 
                                cfg, 
                                scheduler, 
                                qnn, 
                                text_encoder, 
                                vae, 
                                save_dir, 
                                dtype, 
                                model_args, 
                                latent_size, 
                                PRECOMPUTE_TEXT_EMBEDS, 
                                device
                                )

    elif opt.timestep_wise_quant:
        qnn.timestep_wise_quant = True
        
        if not opt.layer_wise_quant and not opt.group_wise_quant and not opt.block_group_wise_quant:
            timestep_set_quant(
                            qnn=qnn,
                            prompts=prompts,
                            cfg=cfg,
                            opt=opt,
                            scheduler=scheduler,
                            text_encoder=text_encoder,
                            vae=vae,
                            dtype=dtype,
                            model_args=model_args,
                            latent_size=latent_size,
                            pre_textemb=PRECOMPUTE_TEXT_EMBEDS,
                            device=device,
                            # weight_quant=True, 
                            # act_quant=False,  
                            )

        elif opt.group_wise_quant:
            qnn.group_wise_quant = True
            qnn.layer_wise_quant = False
            quant_group_list = ['attn', 'cross_attn', 'mlp', 'attn_temp']
            unquant_group_list = ['embedder', 'final', 't_block']
            split_timestep = 4  # split the timesteps into N parts
            n_step = cfg.scheduler.num_sampling_steps
            quant_time_list = [i * (n_step) // split_timestep for i in range(split_timestep)]

            for i in range(len(quant_time_list)):
                for group_name in quant_group_list:
                    qnn.quant_start_t = quant_time_list[split_timestep-i-1]+(n_step//split_timestep-1)
                    qnn.quant_end_t = quant_time_list[split_timestep-i-1]

                    save_dir = os.path.join(opt.save_dir, f"t_{n_step}/{qnn.quant_start_t}_{qnn.quant_end_t}/group/{group_name}")
                    os.makedirs(save_dir, exist_ok=True)

                    qnn.set_quant_state(False, False)

                    qnn.group_name_list = [group_name]
                    qnn.unquant_group_list = unquant_group_list

                    video_generate(
                                    prompts, 
                                    cfg, 
                                    scheduler, 
                                    qnn, 
                                    text_encoder, 
                                    vae, 
                                    save_dir, 
                                    dtype, 
                                    model_args, 
                                    latent_size, 
                                    PRECOMPUTE_TEXT_EMBEDS, 
                                    device
                                    )


        elif opt.layer_wise_quant:
            qnn.layer_wise_quant = True
            qnn.group_wise_quant = False
            with open("./layer_names.txt", 'r', encoding='utf-8') as file:
                quant_layer_list = file.read().splitlines()

            split_timestep = 4  # Uniformly divide the time step into 'split_timestep' parts
            n_step = cfg.scheduler.num_sampling_steps
            quant_time_list = [i * (n_step) // split_timestep for i in range(split_timestep)]

            for i in range(len(quant_time_list)):
                for layer_name in quant_layer_list:
                    qnn.quant_start_t = quant_time_list[split_timestep-i-1]+(n_step//split_timestep-1)
                    qnn.quant_end_t = quant_time_list[split_timestep-i-1]

                    save_dir = os.path.join(opt.save_dir, f"t_{n_step}/{qnn.quant_start_t}_{qnn.quant_end_t}/layer/{layer_name}")
                    os.makedirs(save_dir, exist_ok=True)

                    qnn.set_quant_state(False, False)

                    qnn.quant_layer_name = [layer_name]

                    video_generate(
                                    prompts, 
                                    cfg, 
                                    scheduler, 
                                    qnn, 
                                    text_encoder, 
                                    vae, 
                                    save_dir, 
                                    dtype, 
                                    model_args, 
                                    latent_size, 
                                    PRECOMPUTE_TEXT_EMBEDS, 
                                    device
                                    )


        elif opt.block_group_wise_quant:
            for bit_width in [4,6,8]:
                qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='weight')
                qnn.set_layer_bit(model=qnn, n_bit=bit_width, quant_level='reset', bit_type='act')

                qnn.block_group_wise_quant = True
                qnn.layer_wise_quant = False
                qnn.group_wise_quant = False

                # with open("./layer_names.txt", 'r', encoding='utf-8') as file:
                #     quant_layer_list = file.read().splitlines()

                split_timestep = 4  # Uniformly divide the time step into 'split_timestep' parts
                n_step = cfg.scheduler.num_sampling_steps
                quant_time_list = [i * (n_step) // split_timestep for i in range(split_timestep)]

                # 从txt文件中读取模型层名字
                with open('./t2v/layer_names.txt', 'r') as f:
                    layers = f.read().splitlines()

                group_name = opt.group_quant  ###############################################
                
                for i in range(len(quant_time_list)):
                    for j in range(28):
                        selected_layers = [layer for layer in layers if f'model.blocks.{j}.' in layer]
                        selected_layers_group = [layer for layer in selected_layers if group_name in layer]
                        # selected_layers_list.extend(selected_layers_group)

                        qnn.quant_start_t = quant_time_list[split_timestep-i-1]+(n_step//split_timestep-1)
                        qnn.quant_end_t = quant_time_list[split_timestep-i-1]

                        qnn.quant_layer_name = selected_layers_group
                        
                        save_dir = os.path.join(opt.save_dir, f"bit_{bit_width}/t_{n_step}/{qnn.quant_start_t}_{qnn.quant_end_t}/block_{j}")
                        os.makedirs(save_dir, exist_ok=True)
                        qnn.set_quant_state(False, False)

                        video_generate(
                                        prompts, 
                                        cfg, 
                                        scheduler, 
                                        qnn, 
                                        text_encoder, 
                                        vae, 
                                        save_dir, 
                                        dtype, 
                                        model_args, 
                                        latent_size, 
                                        PRECOMPUTE_TEXT_EMBEDS, 
                                        device
                                        )




if __name__ == "__main__":
    main()
