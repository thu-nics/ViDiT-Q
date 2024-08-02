import argparse
import json
import os
from glob import glob

from mmengine.config import Config
from torch.utils.tensorboard import SummaryWriter


# def parse_args(training=False, get_calib=False, ptq=False, quant_inference=False):
def parse_args(training=False, mode=None):

    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("config", help="model config file path")

    parser.add_argument("--seed", default=42, type=int, help="generation seed")
    parser.add_argument("--ckpt_path", type=str, default=None, required=True, help="path to model ckpt; will overwrite cfg.ckpt_path if specified")
    parser.add_argument("--batch_size", default=None, type=int, help="batch size")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device")
    parser.add_argument("--outdir", default=None, required=True, type=str, help="path to write results to")
    parser.add_argument("--precompute_text_embeds", default=None, type=str, help="path to load the precomputed text_embeds")

    # ======================================================
    # Inference
    # ======================================================

    if not training:
        # prompt
        parser.add_argument("--prompt_path", default=None, type=str, help="path to prompt txt file")
        parser.add_argument("--save_dir", default=None, type=str, help="path to save generated samples, if leave empty, is outdir/generated_videos")
        # hyperparameters
        parser.add_argument("--num_sampling_steps", default=None, type=int, help="sampling steps")
        parser.add_argument("--cfg_scale", default=None, type=float, help="balance between cond & uncond")
        # sampler
        parser.add_argument("--sampler", default="ddim", type=str, help="sampler type")
    else:
        parser.add_argument("--wandb", default=None, type=bool, help="enable wandb")
        parser.add_argument("--load", default=None, type=str, help="path to continue training")
        parser.add_argument("--data_path", default=None, type=str, help="path to data csv")

    # additional args
    if mode == 'get_calib':
        parser.add_argument("--data_num", default=100, type=int)
        parser.add_argument("--save_inp_oup", action="store_true")
    elif mode == 'ptq':
        parser.add_argument("--calib_data", default=None, type=str, help="path to quantization calib data")
    elif mode == "quant_inference":
        parser.add_argument("--dataset_type", default="opensora", type=str)
        parser.add_argument(
            "--quant_ckpt",
            type=str,
            default=None,
            help="path to config which constructs model",
        )
    elif mode == "qat":
        parser.add_argument("--quant_dir", default=None, type=str, help="the dir with quant ckpt and quant config")
    elif mode is None:
        pass
    else:
        raise NotImplementedError

    if mode == 'ptq' or mode == 'quant_inference':
        parser.add_argument(
            "--ptq_config",
            type=str,
            default=None,
            help="path to config which constructs model",
        ) 
        parser.add_argument(
            "--part_quant",
            action="store_true",
            help="whether only quant a part",
        )
        parser.add_argument(
            "--skip_quant_weight",
            action="store_true",
            help="whether to skip weight quantization",
        ) 
        parser.add_argument(
            "--skip_quant_act",
            action="store_true",
            help="whether to skip activation quantization",
        ) 
        parser.add_argument(
            "--num_videos",
            type=int,
            default=1000,
            help="number of generated videos",
        )
        parser.add_argument(
            "--layer_wise_quant",
            action="store_true",
            help="whether only quant a part of layers",
        )
        parser.add_argument(
            "--group_wise_quant",
            action="store_true",
            help="whether only quant a group",
        )
        parser.add_argument(
            "--timestep_wise_quant",
            action="store_true",
            help="whether only quant a part of timesteps",
        )
        parser.add_argument(
            "--block_group_wise_quant",
            action="store_true",
            help="whether only quant a part of timesteps",
        )
        parser.add_argument(
            "--quant_ratio",
            type=float,
            default=1.0,
            help="the ratio of quant layer",
        )
        parser.add_argument(
            "--part_fp",
            action="store_true",
            help="whether only fp a part of layer",
        )
        parser.add_argument(
            "--fp_ratio",
            type=float,
            default=1.0,
            help="the ratio of fp layer",
        )
        parser.add_argument(
            "--timestep_wise_mp",
            action="store_true",
            help="timestep wise mixed precision",
        )
        parser.add_argument(
            "--weight_mp",
            action="store_true",
            help="mixed precision for weight",
        )
        parser.add_argument(
            "--act_mp",
            action="store_true",
            help="mixed precision for act",
        )
        parser.add_argument(
            "--time_mp_config_weight",
            type=str,
            default=None,
            help="path to config of mixed precision for weight",
        )
        parser.add_argument(
            "--time_mp_config_act",
            type=str,
            default=None,
            help="path to config of mixed precision for act",
        )
        parser.add_argument(
            "--group_quant",
            type=str,
            default=None,
            help="path to config of mixed precision for act",
        )
        parser.add_argument(
            "--smooth_quant_alpha",
            nargs='+',
            type=float,
            default=None,
            help="path to config of mixed precision for act",
        )     
        parser.add_argument(
            "--block_wise_quant_progressively",
            action="store_true",
            help="mixed precision for act",
        )
        parser.add_argument(
            "--block_wise_quant",
            action="store_true",
            help="mixed precision for act",
        )
    return parser.parse_args()


# def merge_args(cfg, args, training=False, get_calib=False, ptq=False, quant_inference=False):
def merge_args(cfg, args, training=False, mode=None):
    # some specialized handling of args
    if not hasattr(cfg,"multi_resolution"):
        cfg["multi_resolution"] = False
    if not hasattr(args,"save_dir") or args.save_dir is None:
        args.save_dir = os.path.join(args.outdir,"generated_videos")
    if hasattr(args,"ckpt_path"):
        if args.ckpt_path is not None:
            cfg.model["from_pretrained"] = args.ckpt_path
    if not training:
        if hasattr(args, "cfg_scale"):
            if args.cfg_scale is not None:
                cfg.scheduler["cfg_scale"] = args.cfg_scale

    if mode == "quant_inference":
        # append dataset type to args.save_dir
        args.save_dir = args.save_dir + '_' + args.dataset_type

        default_prompt_path = {
                "opensora": "./t2v/assets/texts/t2v_samples.txt",
                "ucf": "./assets/texts/prompt_ucf.txt",
                }

        # assign default prompt path
        if not hasattr(args, "prompt_path"):
            args.prompt_path = default_prompt_path[args.dataset_type]
        elif args.prompt_path is None:
            args.prompt_path = default_prompt_path[args.dataset_type]

    # INFO: args overwrite the config
    cfg.merge_from_dict({k: v for k,v in vars(args).items() if v is not None})
    return cfg

    # if args.ckpt_path is not None:
        # cfg.model["from_pretrained"] = args.ckpt_path
        # args.ckpt_path = None

    # cfg.gpu = args.gpu
    # cfg.sampler = args.sampler
    # cfg.save_dir = args.save_dir
    # if get_calib:
        # cfg.data_num = args.data_num
        # cfg.save_inp_oup = args.save_inp_oup
    # if ptq:
        # cfg.outdir = args.outdir
        # cfg.ptq_config = args.ptq_config
        # if args.calib_data is not None:  # if have args calib_data, overwrite config
            # cfg.calib_data = args.calib_data
    # if args.prompt_path is not None:
        # cfg.prompt_path = args.prompt_path  # args prompt_path could overwrite config

    # if quant_inference:
        # cfg.outdir = args.outdir
        # cfg.ptq_config = args.ptq_config
        # cfg.quant_ckpt = args.quant_ckpt
        # cfg.skip_quant_weight = args.skip_quant_weight
        # cfg.skip_quant_act = args.skip_quant_act
        # cfg.num_videos = args.num_videos
        
    # if ptq or quant_inference:
        # cfg.part_quant = args.part_quant
        # cfg.part_fp = args.part_fp
        # cfg.quant_ratio = args.quant_ratio
        # cfg.fp_ratio = args.fp_ratio
        
    # if not training:
        # if args.cfg_scale is not None:
            # cfg.scheduler["cfg_scale"] = args.cfg_scale
            # args.cfg_scale = None

    # if "multi_resolution" not in cfg:
        # cfg["multi_resolution"] = False


    # return cfg


def parse_configs(training=False, mode=None):
    args = parse_args(training, mode)
    cfg = Config.fromfile(args.config)
    cfg = merge_args(cfg, args, training, mode)
    return cfg


def create_experiment_workspace(cfg):
    """
    This function creates a folder for experiment tracking.

    Args:
        args: The parsed arguments.

    Returns:
        exp_dir: The path to the experiment folder.
    """
    # Make outputs folder (holds all experiment subfolders)
    os.makedirs(cfg.outputs, exist_ok=True)
    experiment_index = len(glob(f"{cfg.outputs}/*"))

    # Create an experiment folder
    model_name = cfg.model["type"].replace("/", "-")
    exp_name = f"{experiment_index:03d}-F{cfg.num_frames}S{cfg.frame_interval}-{model_name}"
    exp_dir = f"{cfg.outputs}/{exp_name}"
    os.makedirs(exp_dir, exist_ok=True)
    return exp_name, exp_dir


def save_training_config(cfg, experiment_dir):
    with open(f"{experiment_dir}/config.txt", "w") as f:
        json.dump(cfg, f, indent=4)


def create_tensorboard_writer(exp_dir):
    tensorboard_dir = f"{exp_dir}/tensorboard"
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(tensorboard_dir)
    return writer
