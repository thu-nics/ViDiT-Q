import logging
from typing import Union
import numpy as np
from tqdm import trange

import torch
import torch.nn as nn
import torch.nn.functional as F

from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_block import BaseQuantBlock
from qdiff.models.quant_model import QuantModel
from qdiff.quantizer.base_quantizer import BaseQuantizer, lp_loss

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


class LossFunction:
    '''Wrapper of LossFunc, Get the round_loss and reconstruction_loss'''
    def __init__(self,
                 module,
                 round_loss_type: str = 'relaxation',
                 reconstruction_loss_type: str = 'mse',
                 lambda_coeff: float = 1.,  # the coeff between two loss
                 iters: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.,
                 module_type='layer',
                 use_reconstruction_loss=False,
                 use_round_loss=False,
                 ):

        self.module = module
        self.module_type = module_type
        self.round_loss_type = round_loss_type
        self.reconstruction_loss_type = reconstruction_loss_type
        self.lambda_coeff = lambda_coeff
        self.loss_start = iters * warmup
        self.warmup = warmup
        self.iters = iters
        self.p = p
        self.use_reconstruction_loss = use_reconstruction_loss
        self.use_round_loss = use_round_loss

        self.temp_decay = LinearTempDecay(iters, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None):
        """
        Compute the total loss for adaptive rounding:
        reconstruction_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        total_loss = 0.

        self.count += 1
        if self.use_reconstruction_loss:
            if self.reconstruction_loss_type == 'mse':
                reconstruction_loss = lp_loss(pred, tgt, p=int(self.p), reduction='all')
            elif self.reconstruction_loss_type == 'fisher_diag':
                reconstruction_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
            elif self.reconstruction_loss_type == 'fisher_full':
                a = (pred - tgt).abs()
                grad = grad.abs()
                batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
                reconstruction_loss = (batch_dotprod * a * grad).mean() / 100
            else:
                raise ValueError('Not supported reconstruction loss function: {}'.format(self.reconstruction_loss_type))
        else:
            reconstruction_loss = 0.

        b = self.temp_decay(self.count)
        if self.use_round_loss:
            if self.count < self.loss_start or self.round_loss_type == 'none':
                b = round_loss = 0
            elif self.round_loss_type == 'relaxation':
                round_loss = 0
                # DEBUG: didnot consider split rounding error
                if self.module_type == 'layer':
                    round_vals = self.module.weight_quantizer.get_soft_targets()
                    round_loss += self.lambda_coeff * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
                elif self.module_type == 'block':
                    round_loss = 0
                    for name, module_ in self.module.named_modules():
                        if isinstance(module_, QuantLayer):
                            round_vals = module_.weight_quantizer.get_soft_targets()
                            round_loss += self.lambda_coeff * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            else:
                raise NotImplementedError
        else:
            round_loss = 0.

        total_loss += reconstruction_loss
        total_loss += round_loss
        if self.count % 100 == 0:
            reconstruction_loss = -1 if not self.use_reconstruction_loss else reconstruction_loss
            round_loss = -1 if not self.use_round_loss else round_loss
            logger.info('Total loss:\t{:.6f} (rec:{:.6f}, round:{:.6})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(reconstruction_loss), float(round_loss), b, self.count))
        return total_loss


class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))

def prepare_coco_text_and_image(json_file):
    info = json.load(open(json_file, 'r'))
    annotation_list = info["annotations"]
    image_caption_dict = {}
    for annotation_dict in annotation_list:
        if annotation_dict["image_id"] in image_caption_dict.keys():
            image_caption_dict[annotation_dict["image_id"]].append(annotation_dict["caption"])
        else:
            image_caption_dict[annotation_dict["image_id"]] = [annotation_dict["caption"]]
    captions = list(image_caption_dict.values())
    image_ids = list(image_caption_dict.keys())

    active_captions = []
    for texts in captions:
        active_captions.append(texts[0])

    image_paths = []
    for image_id in image_ids:
        image_paths.append("/share/public/diffusion_quant/coco/coco/val2014/"+f"COCO_val2014_{image_id:012}.jpg")
    return active_captions, image_paths

# ---------- save input output activation & grad ---------------------
def save_in_out_data(model: QuantModel, layer: Union[QuantLayer, BaseQuantBlock], calib_data: torch.Tensor, config, model_type='sdxl', split_save_attn=False):
    # asym: bool = False, act_quant: bool = False, batch_size: int = 32, keep_gpu: bool = True,
                      # cond: bool = True, split_save_attn: bool = False, model_type='sdxl'):
    """
    Save input data and output data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantLayer or QuantBlock
    :param calib_data: calibration data set
    :param asym: if Ture, save quantized input and full precision output
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :param cond: conditional generation or not
    :param split_save_attn: avoid OOM when caching n^2 attention matrix when n is large
    :return: input and output data
    """
    device = next(model.parameters()).device
    get_in_out = GetLayerInOut(model, layer, model_type=model_type, previous_layer_quantized=True)
    cached_batches = []
    cached_inps, cached_outs = None, None
    torch.cuda.empty_cache()

    assert config.conditional # only support conditional generation
    assert not split_save_attn, "not checked for now"

    if model_type == 'sdxl':
        calib_xs, calib_ts, calib_conds, calib_added_conds = calib_data
        calib_added_text_embeds = calib_added_conds["text_embeds"]
        calib_added_time_ids = calib_added_conds["time_ids"]
        calib_masks = None
    elif model_type == 'sd':
        calib_xs, calib_ts, calib_conds = calib_data
        calib_masks = None
    elif model_type == 'pixart':
        calib_xs, calib_ts, calib_conds, calib_masks = calib_data
    else:
        raise NotImplementedError

    # INO: whether split attention map to avoid OOM
    # if split_save_attn:
    #     logger.info("Checking if attention is too large...")

    #     if model_type == 'sdxl':
    #         calib_added_conds["text_embeds"] = calib_added_text_embeds[:1].to(device)
    #         calib_added_conds["time_ids"] = calib_added_time_ids[:1].to(device)
    #     test_inp, test_out = get_in_out(
    #         calib_xs[:1].to(device),
    #         calib_ts[:1].to(device),
    #         calib_conds[:1].to(device),
    #         calib_added_conds,
    #     )

    #     split_save_attn = False
    #     if (isinstance(test_inp, tuple) and test_inp[0].shape[1] == test_inp[0].shape[2]):
    #         logger.info(f"test_inp shape: {test_inp[0].shape}, {test_inp[1].shape}")
    #         if test_inp[0].shape[1] == 4096:
    #             split_save_attn = True
    #     if test_out.shape[1] == test_out.shape[2]:
    #         logger.info(f"test_out shape: {test_out.shape}")
    #         if test_out.shape[1] == 4096:
    #             split_save_attn = True

    #     if split_save_attn:
    #         logger.info("Confirmed. Trading speed for memory when caching attn matrix calibration data")
    #         inds = np.random.choice(calib_xs.size(0), calib_xs.size(0) // 2, replace=False)
    #     else:
    #         logger.info("Nope. Using normal caching method")

    batch_size = config.calib_data.batch_size
    iters = int(calib_xs.size(0) / batch_size)
    l_in_0, l_in_1, l_in, l_out = 0, 0, 0, 0
    if split_save_attn:
        num //= 2

    # INFO: iter through all the calib_data, save all input and output
    # defaults
    calib_masks_ = None
    tmp_kwargs = {}
    calib_added_conds = {}
    iters = 16  # DEBUG_ONLY: not using the whole calib data
    for i in trange(iters):
        if model_type == 'sdxl':
            calib_added_conds["text_embeds"] = calib_added_text_embeds[i * batch_size:(i + 1) * batch_size].to(device)
            calib_added_conds["time_ids"] = calib_added_time_ids[i * batch_size:(i + 1) * batch_size].to(device)
        elif model_type == 'pixart':
            calib_masks_ = calib_masks[i * batch_size:(i + 1) * batch_size].to(device)
            tmp_kwargs = {
                'fcf': 4.5,
                'data_info': {
                    'img_hw': torch.tensor([[1024., 1024.]], device=device),
                    'aspect_ratio': torch.tensor([[1.]], device=device)
                }
            }
        else:
            pass

        cur_inp, cur_out = get_in_out(
            calib_xs[i * batch_size:(i + 1) * batch_size].to(device),
            calib_ts[i * batch_size:(i + 1) * batch_size].to(device),
            calib_conds[i * batch_size:(i + 1) * batch_size].to(device),
            calib_added_conds,
            calib_masks_,
        )
        if isinstance(cur_inp, tuple):
            if(len(cur_inp)>2):
                cur_inp = list(cur_inp)
                for i in range(len(cur_inp)):
                    cur_inp[i] = cur_inp[i].cpu() if cur_inp[i] is not None else None  # difference
                cached_batches.append((tuple(cur_inp), cur_out.cpu()))
            else:
                cur_x, cur_t = cur_inp
                if not split_save_attn:
                    cached_batches.append(((cur_x.cpu(), cur_t.cpu()), cur_out.cpu()))
                else:
                    if cached_inps is None:
                        l_in_0 = cur_x.shape[0] * num
                        l_in_1 = cur_t.shape[0] * num
                        cached_inps = [torch.zeros(l_in_0, *cur_x.shape[1:]), torch.zeros(l_in_1, *cur_t.shape[1:])]
                    cached_inps[0].index_copy_(0, torch.arange(i * cur_x.shape[0], (i + 1) * cur_x.shape[0]), cur_x.cpu())
                    cached_inps[1].index_copy_(0, torch.arange(i * cur_t.shape[0], (i + 1) * cur_t.shape[0]), cur_t.cpu())
        else:
            if not split_save_attn:
                cached_batches.append((cur_inp.cpu(), cur_out.cpu()))
            else:
                if cached_inps is None:
                    l_in = cur_inp.shape[0] * num
                    cached_inps = torch.zeros(l_in, *cur_inp.shape[1:])
                cached_inps.index_copy_(0, torch.arange(i * cur_inp.shape[0], (i + 1) * cur_inp.shape[0]), cur_inp.cpu())

        # if split_save_attn:
        #     if cached_outs is None:
        #         l_out = cur_out.shape[0] * num
        #         cached_outs = torch.zeros(l_out, *cur_out.shape[1:])
        #     cached_outs.index_copy_(0, torch.arange(i * cur_out.shape[0], (i + 1) * cur_out.shape[0]), cur_out.cpu())

    if not split_save_attn:
        if isinstance(cached_batches[0][0], tuple):
            # if input_type in tuple, QuantTransformerBlock
            if len(cached_batches[0][0]) > 2:
                cached_inps = []
                for i in range(len(cached_batches[0][0])):
                    if cached_batches[0][0][i] == None:
                        cached_inps.append(None)
                    else:
                        cached_inps.append(torch.cat([x[0][i] for x in cached_batches]))  # difference
            else:
                cached_inps = [
                    torch.cat([x[0][0] for x in cached_batches]),
                    torch.cat([x[0][1] for x in cached_batches])
                ]
        else:
            cached_inps = torch.cat([x[0] for x in cached_batches])
            cached_outs = torch.cat([x[1] for x in cached_batches])

    if isinstance(cached_inps, list):
        for i in range(len(cached_inps)):
            logger.info(f"in {i} shape: {cached_inps[i].shape}") if cached_inps[i] is not None else logger.info(f"in {i} : None") 
    else:
        logger.info(f"in shape: {cached_inps.shape}")
    logger.info(f"out shape: {cached_outs.shape}")
    torch.cuda.empty_cache()

    # INFO: move data to gpu, why does it need to move to cpu at first?
    if isinstance(cached_inps, list):
        if len(cached_inps)==7:
            cached_inps[0] = cached_inps[0].to(device)
            cached_inps[2] = cached_inps[2].to(device)
        else:
            cached_inps[0] = cached_inps[0].to(device)
            cached_inps[1] = cached_inps[1].to(device)
    else:
        cached_inps = cached_inps.to(device)
    cached_outs = cached_outs.to(device)

    return cached_inps, cached_outs

def save_grad_data(model: QuantModel, layer: Union[QuantLayer, BaseQuantBlock], calib_data: torch.Tensor,
                   damping: float = 1., act_quant: bool = False, batch_size: int = 32,
                   keep_gpu: bool = True):
    """
    Save gradient data of a particular layer/block over calibration dataset.

    :param model: QuantModel
    :param layer: QuantLayer or QuantBlock
    :param calib_data: calibration data set
    :param damping: damping the second-order gradient by adding some constant in the FIM diagonal
    :param act_quant: use activation quantization
    :param batch_size: mini-batch size for calibration
    :param keep_gpu: put saved data on GPU for faster optimization
    :return: gradient data
    """
    device = next(model.parameters()).device
    get_grad = GetLayerGrad(model, layer, device, act_quant=act_quant)
    cached_batches = []
    torch.cuda.empty_cache()

    for i in range(int(calib_data[0].size(0) / batch_size)):
        cur_grad = get_grad(calib_data[0][i * batch_size:(i + 1) * batch_size])
        cached_batches.append(cur_grad.cpu())

    cached_grads = torch.cat([x for x in cached_batches])
    cached_grads = cached_grads.abs() + 1.0
    # scaling to make sure its mean is 1
    # cached_grads = cached_grads * torch.sqrt(cached_grads.numel() / cached_grads.pow(2).sum())
    torch.cuda.empty_cache()
    if keep_gpu:
        cached_grads = cached_grads.to(device)
    return cached_grads


class StopForwardException(Exception):
    """
    Used to throw and catch an exception to stop traversing the graph
    """
    pass


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
            raise StopForwardException


class GetLayerInOut:
    def __init__(self, model: QuantModel, layer: Union[QuantLayer, BaseQuantBlock], model_type='sd', previous_layer_quantized=False):
                 # device: torch.device, asym: bool = False, act_quant: bool = False, model_type='sd'):
        self.model = model
        self.layer = layer
        self.previous_layer_quantized = previous_layer_quantized
        # self.device = device
        # self.act_quant = act_quant
        self.model_type = model_type
        self.data_saver = DataSaverHook(store_input=True, store_output=True, stop_forward=True)

    def __call__(self, x, timesteps, context=None, added_conds=None, mask=None):

        self.model.eval()  # temporarily use eval mode
        # INFO: save the quant_state, since it will be written by (False, False)
        model_quant_weight, model_quant_act = self.model.get_quant_state()
        layer_quant_weight, layer_quant_act = self.layer.get_quant_state()
        self.model.set_quant_state(False, False)  # use all FP model
        handle = self.layer.register_forward_hook(self.data_saver)

        with torch.no_grad():
            # for pixart
            if self.model_type == 'pixart':
                tmp_kwargs = {
                    'fcf': 4.5,
                    'data_info': {
                        'img_hw': torch.tensor([[1024., 1024.]], device='cuda'),
                        'aspect_ratio': torch.tensor([[1.]], device='cuda')
                    }
                }
            else:
                tmp_kwargs = {}

            try:
                _ = self.model(x, timesteps, context, added_cond_kwargs=added_conds, mask=mask, **tmp_kwargs)
            except StopForwardException:
                pass

            if self.previous_layer_quantized:
                # INFO: rewrite the input data, with *all previous layer* quantized
                # overwrite input with network quantized
                self.data_saver.store_output = False  # avoid overwrite the output data
                self.model.set_quant_state(model_quant_weight, model_quant_act)  # restore original model quant_state
                try:
                    _ = self.model(x, timesteps, context, added_cond_kwargs=added_conds, mask=mask, **tmp_kwargs)
                except StopForwardException:
                    pass
                self.data_saver.store_output = True
        handle.remove()

        self.model.set_quant_state(model_quant_weight, model_quant_act)
        self.layer.set_quant_state(layer_quant_weight, layer_quant_act)
        self.model.train()

        if len(self.data_saver.input_store) > 1 and len(self.data_saver.input_store) < 7 and torch.is_tensor(self.data_saver.input_store[1]):
            return (self.data_saver.input_store[0].detach(),
                self.data_saver.input_store[1].detach()), self.data_saver.output_store.detach()
        elif len(self.data_saver.input_store) == 7:
            input_tuple = []
            for input in self.data_saver.input_store:
                if input == None:
                    input_tuple.append(input)
                else:
                    input_tuple.append(input.detach())
            return tuple(input_tuple), self.data_saver.output_store.detach()  # difference
        else:
            return self.data_saver.input_store[0].detach(), self.data_saver.output_store.detach()

class GradSaverHook:
    def __init__(self, store_grad=True):
        self.store_grad = store_grad
        self.stop_backward = False
        self.grad_out = None

    def __call__(self, module, grad_input, grad_output):
        if self.store_grad:
            self.grad_out = grad_output[0]
        if self.stop_backward:
            raise StopForwardException


class GetLayerGrad:
    def __init__(self, model: QuantModel, layer: Union[QuantLayer, BaseQuantBlock],
                 device: torch.device, act_quant: bool = False):
        self.model = model
        self.layer = layer
        self.device = device
        self.act_quant = act_quant
        self.data_saver = GradSaverHook(True)

    def __call__(self, model_input):
        """
        Compute the gradients of block output, note that we compute the
        gradient by calculating the KL loss between fp model and quant model

        :param model_input: calibration data samples
        :return: gradients
        """
        self.model.eval()

        handle = self.layer.register_backward_hook(self.data_saver)
        with torch.enable_grad():
            try:
                self.model.zero_grad()
                inputs = model_input.to(self.device)
                self.model.set_quant_state(False, False)
                out_fp = self.model(inputs)
                quantize_model_till(self.model, self.layer, self.act_quant)
                out_q = self.model(inputs)
                loss = F.kl_div(F.log_softmax(out_q, dim=1), F.softmax(out_fp, dim=1), reduction='batchmean')
                loss.backward()
            except StopForwardException:
                pass

        handle.remove()
        self.model.set_quant_state(False, False)
        self.layer.set_quant_state(True, self.act_quant)
        self.model.train()
        return self.data_saver.grad_out.data


def quantize_model_till(model: QuantLayer, layer: Union[QuantLayer, BaseQuantBlock], act_quant: bool = False):
    """
    We assumes modules are correctly ordered, holds for all models considered
    :param model: quantized_model
    :param layer: a block or a single layer.
    """
    model.set_quant_state(False, False)
    for name, module in model.named_modules():
        if isinstance(module, (QuantLayer, BaseQuantBlock)):
            module.set_quant_state(True, act_quant)
        if module == layer:
            break
