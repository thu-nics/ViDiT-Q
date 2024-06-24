import logging
from typing import Union
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

def get_quant_calib_data(config, sample_data, custom_steps=None, model_type='opensora', repeat_interleave=False):
    num_samples, num_st = config.calib_data.n_samples, custom_steps
    nsteps = len(sample_data["ts"])
    assert(nsteps >= custom_steps)  # custom_steps subsample the calib data
    if len(sample_data["ts"][0].shape) == 0:  # expand_dim for 0-dim tensor
        for i in range(nsteps):
            sample_data["ts"][i] = sample_data["ts"][i][None]

    # INFO: preprocess the batch-dim for CFG
    # sample_data has [2(cond & uncond), bs] layout
    # however, the ptq and quant_infer, we use batch size to index them
    # we need to permute it back into [bs,2] for batch choice (for QNN infer in PTQ)

    # for key in sample_data:
        # # shift back the dimension
        # shape_ = list(sample_data[key].shape)
        # sample_data[key] = sample_data[key].reshape([shape_[0]]+[2,shape_[1]//2]+shape_[2:])
        # sample_data[key] = sample_data[key].permute([0,2,1,*range(3,len(shape_)+1)])
        # sample_data[key] = sample_data[key].reshape(shape_)

    if not repeat_interleave:
        timesteps = list(range(0, nsteps, nsteps//num_st))
        logger.info(f'Selected {len(timesteps)} steps from {nsteps} sampling steps')

        xs_lst = [sample_data["xs"][i][:num_samples*2] for i in timesteps]
        ts_lst = [sample_data["ts"][i][:num_samples*2] for i in timesteps]
        cond_emb_lst = [sample_data["cond_emb"][i][:num_samples*2] for i in timesteps]
        mask_lst = [sample_data["mask"][i][:num_samples*2] for i in timesteps]
    else:
        ts_downsample_rate = nsteps // num_steps_chosen
        logger.info(f'Selected {len(timesteps)} steps from {nsteps} sampling steps')

        # INFO: classifier free guidance, have 2x for each sample
        xs_lst = [sample_data["xs"][i][:num_samples*2] for i in range(nsteps)]
        ts_lst = [sample_data["ts"][i][:num_samples*2] for i in timesteps]
        cond_emb_lst = [sample_data["cond_emb"][i][:num_samples*2] for i in range(nsteps)]
        mask_lst = [sample_data["mask"][i][:num_samples*2] for i in range(nsteps)]

    xs = torch.cat(xs_lst, dim=0)
    ts = torch.cat(ts_lst, dim=0)
    cond_embs = torch.cat(cond_emb_lst, dim=0)
    masks = torch.cat(mask_lst, dim=0)

    if model_type == 'opensora' or model_type == 'pixart':
        return xs, ts, cond_embs, masks
    else:
        raise NotImplementedError
    
@torch.no_grad()
def load_quant_params(qnn, ckpt_path, dtype=torch.float32):
    print("Loading quantized model checkpoint")
    ckpt = torch.load(ckpt_path, map_location='cpu')
    qnn.set_module_name_for_quantizer(module=qnn.model)
    qnn.set_quant_params_dict(ckpt, dtype=dtype)

class DataSaverHook:
    """
    Forward hook that stores the input and output of a block
    """
    def __init__(self, store_input=False, store_output=False, stop_forward=False):
        self.store_input = store_input
        self.store_output = store_output
        self.stop_forward = stop_forward

        self.input_store = None
        self.output_store = None

    def __call__(self, module, input_batch, output_batch):
        if self.store_input:
            self.input_store = input_batch 
        if self.store_output:
            self.output_store = output_batch 
        if self.stop_forward:
            import ipdb; ipdb.set_trace()
            raise StopForwardException


