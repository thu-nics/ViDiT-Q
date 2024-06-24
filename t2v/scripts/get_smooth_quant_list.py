# import (same as inference.py)
import os
import sys
# sys.path.append(".")

import torch
import colossalai
import torch.distributed as dist
from mmengine.runner import set_random_seed

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.build_model import build_models, build_models_only
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from colossalai.cluster import DistCoordinator

from mmengine.config import Config

import random
import math
import torch.nn as nn
from qdiff.utils import DataSaverHook

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def select_calib_data(calib_data, data_num, select_type="random", t=None, t_start=None, t_end=None, device="cuda"):
    num_step, bs = calib_data["xs"].shape[0], calib_data["xs"].shape[1]
    if select_type == "random":
        if num_step * bs < data_num:
            data_num = num_step * bs
        indices = []
        while len(indices) < data_num:
            pair = (random.randint(0, num_step - 1),
                    random.randint(0, bs - 1))
            if pair not in indices:
                indices.append(pair)
    elif select_type == "timestep":
        if 1 * bs < data_num:
            data_num = 1 * bs
        indices = []
        while len(indices) < data_num:
            pair = (t, random.randint(0, bs - 1))
            if pair not in indices:
                indices.append(pair)
    elif select_type == "timestep_range":
        if (t_end - t_start) * bs < data_num:
            data_num = t_end - t_start
        indices = []
        while len(indices) < data_num:
            pair = (random.randint(t_start, t_end - 1), random.randint(0, bs - 1))
            if pair not in indices:
                indices.append(pair)
    # TODO：comparison between different channel
    else:
        raise NotImplementedError

    selected_data = {}
    for key in calib_data.keys():
        selected_data[key] = torch.stack([calib_data[key][index] for index in indices], dim=0).to(device)
        
    return selected_data

def get_data(model, module, input_data):
    data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=False)
    handle = module.register_forward_hook(data_saver) 

    with torch.no_grad():
        xs, ts, cond_emb, mask = input_data["xs"], input_data["ts"], input_data["cond_emb"], input_data["mask"]
        print(f"Getting the intermediate output at one inference...Input shape: ")
        print(f"xs:{xs.shape}")
        print(f"ts:{ts.shape}")
        print(f"cond_emb:{cond_emb.shape}")
        _ = model(xs, ts, cond_emb, mask=mask)

    handle.remove()
    return data_saver.input_store, data_saver.output_store

def get_data_list(model, module_dict, input_data):
    data_savers = []
    handles = []
    module_names = []
    for module_name, module in module_dict.items():
        data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=False)
        handle = module.register_forward_hook(data_saver) 
        data_savers.append(data_saver)
        handles.append(handle)
        module_names.append(module_name)

    with torch.no_grad():
        xs, ts, cond_emb, mask = input_data["xs"], input_data["ts"], input_data["cond_emb"], input_data["mask"]
        print(f"Getting the intermediate output at one inference...Input shape: ")
        print(f"xs:{xs.shape}")
        print(f"ts:{ts.shape}")
        print(f"cond_emb:{cond_emb.shape}")
        _ = model(xs, ts, cond_emb, mask=mask)

    data_input = {}
    data_output = {}

    for i in range(len(module_names)):
        handles[i].remove()
        data_input[module_names[i]] = data_savers[i].input_store
        data_output[module_names[i]] = data_savers[i].output_store
    return data_input, data_output

def get_module_by_name(model, module_name):
    for name, module in model.named_modules():
        if module_name == name:
            return module
        
def cal_diff_wrong(act, weight):
    # quant difficulty of weights. It seems that this metric is wrong
    shift_weight = weight - weight.min(dim=1)[0][:, None]
    diff = shift_weight.max(dim=1)[0] / shift_weight.mean(dim=1)
    diff_weight = diff.mean().item()
    
    # quant difficulty of activation. It seems that this metric is wrong
    shift_act = act - act.min(dim=-1)[0][:, :, None]
    diff = shift_act.max(dim=-1)[0] / shift_act.mean(dim=-1)
    diff_act = diff.mean().item()
    
    return diff_act / diff_weight

def cal_aggregation(x, delta, zero_point, bit_width, sym=False):
    # calculate the aggregation of the data after quantization
    
    n_levels = 2 ** bit_width if not sym else 2 ** (bit_width - 1) - 1
    x_int = round_ste(x / delta) + zero_point
    x_quant = torch.clamp(x_int, 0, n_levels - 1)

    # import ipdb; ipdb.set_trace()
    x_dequant = (x_quant - zero_point) * delta
    
    entropy = 0
    for i in range(0, n_levels):
        p = ((x_quant == i).sum() / x_quant.numel()).item()
        entropy += -p * math.log(max(p, 1e-5))
        
    return entropy

def cal_after_smooth_quant_weights(act, weight, alpha):
    channel_wise_scale = act.abs().max(dim=-2)[0].pow(alpha).mean(dim=0, keepdim=True) / weight.abs().max(dim=0)[0].pow(1 - alpha)
    new_act = act / channel_wise_scale
    new_weight = weight * channel_wise_scale
    
    return new_act, new_weight

def main():
    config_path = "./configs/opensora/inference/16x512x512.py"
    cfg = Config.fromfile(config_path)
    cfg.gpu = "5"
    cfg.model["from_pretrained"] = "/share/liuenshu/temp_files/checkpoints/video/OpenSora-v1-HQ-16x512x512.pth"

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    
    model = build_models_only(cfg, device, dtype, enable_sequence_parallelism=False)
    
    calib_data = torch.load("/share/liuenshu/temp_files/video_exp/inp_oup_data/calib_data.pt")
    for key in calib_data.keys():
        print(f"{key}:{calib_data[key].shape}")

    # get all quant layer
    unquant_list = ["x_embedder", "t_block", "t_embedder", "y_embedder", "final_layer"]
    module_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for unquant in unquant_list:
                if unquant in name:
                    break
            else:
                module_dict[name] = module

    data_num = 10
    t_start = 0
    t_end = 100
    input_data = select_calib_data(calib_data, data_num, select_type="timestep_range", t_start=t_start, t_end=t_end, device=device)
    data_input, data_output = get_data_list(model, module_dict, input_data)
    
    part_fp_list = "/share/liuenshu/Open-Sora/configs/quant/remain_fp.txt"
    with open(part_fp_list,'r') as f:
        lines = f.readlines()
    fp_layer_list = [line.strip() for line in lines]  # strip the '\n'
    
    quant_weight = torch.load("/share/liuenshu/temp_files/video_exp/w4a8/ckpt.pth")
    
    difficulty_diff = {}
    weight_difficulty = {}
    alpha_candidates = [1.0, 1.1, 1.2, 1.3]
    for key in data_input.keys():
        for layer in fp_layer_list:
            if layer in key:
                break
        else:
            module = get_module_by_name(model, key)
            act = data_input[key][0].cpu()
            weight = module.weight.data.cpu()
            in_channel = weight.shape[1]
            assert act.shape[-1] == in_channel
            
            quant = quant_weight[key + ".weight_quantizer"][0]
            delta = quant["delta_list"][0].cpu() # [4,6,8]
            zero_point = quant["zero_point_list"][0].cpu()
            entropy = cal_aggregation(weight, delta, zero_point, bit_width=4, sym=False)
            weight_difficulty[key] = entropy
            
            # calculate the difficulty ratio between weight and activation
            # difficulty_diff[key] = {"fp16_diff": cal_diff_wrong(act=act, weight=weight)}
            # # get the best alpha 
            # if "fc1" in key or "fc2" in key:
            #     min_dis = 1e5
            #     for alpha in alpha_candidates:
            #         new_act, new_weight = cal_after_smooth_quant_weights(act, weight, alpha)
            #         diff = cal_diff_wrong(act=new_act, weight=new_weight)
            #         if abs(diff - 1) < min_dis:
            #             min_alpha = alpha
            #             min_diff = diff
            #             min_dis = abs(diff - 1)
            #     difficulty_diff[key]["best_alpha"] = min_alpha
            #     difficulty_diff[key]["best_diff"] = min_diff
            # else:
            #     difficulty_diff[key]["best_alpha"] = 0.7
            #     difficulty_diff[key]["best_diff"] = None
            
    
    # difficulty_diff = sorted(difficulty_diff.items(), key=lambda x: x[1]["fp16_diff"])
    # torch.save(difficulty_diff, f"/share/liuenshu/temp_files/video_exp/sensitivity/time_{t_start}_{t_end}.pth")    
    import ipdb; ipdb.set_trace()
    
if __name__=="__main__":
    main()