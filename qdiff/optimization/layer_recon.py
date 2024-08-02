import torch
# import linklink as link
import logging
from qdiff.quantizer.base_quantizer import lp_loss
from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_model import QuantModel
from qdiff.quantizer.base_quantizer import StraightThrough
# from qdiff.quantizer.adaptive_rounding import AdaRoundQuantizer
from qdiff.utils import save_grad_data, save_in_out_data, LossFunction

import time

logger = logging.getLogger(__name__)

def layer_reconstruction(model: QuantModel, layer: QuantLayer, calib_data: torch.Tensor, config, param_types, opt_target):
                         # batch_size: int = 32, iters: int = 20000, weight: float = 0.001, opt_mode: str = 'mse',
                         # asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         # warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         # multi_gpu: bool = False, cond: bool = False, is_sm: bool = False):
    """
    Block reconstruction to optimize the output from each layer.

    :param model: QuantModel
    :param layer: QuantLayer that needs to be optimized
    :param calib_data: data for calibration, typically 1024 training images, as described in AdaRound
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    :param multi_gpu: use multi-GPU or not, if enabled, we should sync the gradients
    :param cond: conditional generation or not
    :param is_sm: avoid OOM when caching n^2 attention matrix when n is large
    """

    device = model.device
    batch_size = config.calib_data.batch_size
    if len(calib_data) ==4:
        if config.model.model_type == 'pixart':
            # pixart has {xs, t, condition, masks}
            cached_inps, cached_outs = save_in_out_data(model, layer, calib_data, config, model_type='pixart')
        elif config.model.model_type == 'sdxl':
           # INFO: sdxl has more input {latents, t, condition, +added_cond}
            cached_inps, cached_outs = save_in_out_data(model, layer, calib_data, config, model_type='sdxl')
        else:
            raise NotImplementedError
    else:
        assert config.model.model_type == 'sd'
        cached_inps, cached_outs = save_in_out_data(model, layer, calib_data, config, model_type='sd')
        # cached_inps, cached_outs = save_in_out_data(
            # model, layer, calib_data, asym, act_quant, batch_size, keep_gpu=False, cond=cond, is_sm=is_sm)
    cached_inps, cached_outs = cached_inps.to(device), cached_outs.to(device)

    # INFO: get the grad (not supported)
    if opt_target == 'weight_and_activation':
        use_grad = config.quant.weight.optimization.use_grad
    else:
        use_grad = getattr(config.quant, opt_target).optimization.use_grad
    assert not use_grad, "not supported for now"
    if not use_grad:
        cached_grads = None
    else:
        # INFO: does not support for now
        raise NotImplementedError
        cached_grads = save_grad_data(model, layer, calib_data, act_quant=False, batch_size=batch_size)  # TODO: reduce act_quant
        cached_grads = cached_grads.to(device)

    # INFO: set the quant states, moved within GetLayerInput
    # model_quant_weight, model_quant_act = model.get_quant_state()
    # layer_quant_weight, layer_quant_act = layer.get_quant_state()
    # model.set_quant_state(False, False)

    # INFO: setup quant_params and optimizer, use independent lr for each param group
    opt_params = []  # the param group
    param_group_names = []
    if opt_target == 'weight_and_activation':
        # INFO: should have both of the param groups
        for param_type in param_types['weight']:
            name_ = f"weight.{param_type}"
            param_group_names.append(name_)
            params_ = [getattr(layer.weight_quantizer, param_type)]
            if layer.split != 0:
                params_ += [getattr(layer.weight_quantizer_0, param_type)]
            opt_params += [{
                    'params': params_,
                    'lr': getattr(config.quant.weight.optimization.params, param_type).lr,
                    }]
        for param_type in param_types['activation']:
            name_ = f"activation.{param_type}"
            param_group_names.append(name_)
            params_ = [getattr(layer.act_quantizer, param_type)]
            if layer.split != 0:
                params_ = [getattr(layer.act_quantizer_0, param_type)]
            opt_params += [{
                    'params': params_,
                    'lr': getattr(config.quant.activation.optimization.params, param_type).lr,
                    }]
    elif opt_target in ['weight','activation']:
        for param_type in param_types:
            if opt_target == 'weight':
                name_ = f"weight.{param_type}"
                param_group_names.append(name_)
                params_ = [getattr(layer.weight_quantizer, param_type)]
                if layer.split != 0:
                    params_ += [getattr(layer.weight_quantizer_0, param_type)]
                opt_params += [{
                        'params': params_,
                        'lr': getattr(config.quant.weight.optimization.params, param_type).lr,
                        }]
            elif opt_target == 'activation':
                name_ = f"activation.{param_type}"
                param_group_names.append(name_)
                params_ = [getattr(layer.act_quantizer, param_type)]
                if layer.split != 0:
                    params_ = [getattr(layer.act_quantizer_0, param_type)]
                opt_params += [{
                        'params': params_,
                        'lr': getattr(config.quant.activation.optimization.params, param_type).lr,
                        }]
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(opt_params)

    if opt_target == 'weight_and_activation':
        iters = config.quant.weight.optimization.iters
        assert config.quant.weight.optimization.iters == config.quant.activation.optimization.iters
    else:
        iters = getattr(config.quant,opt_target).optimization.iters
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)
    scheduler = None

    # INFO: unpack the config for loss 
    if opt_target == 'weight_and_activation':
        logging.info("When joint optimization, use weight's quant config")
        config_loss = config.quant.weight.optimization.loss
        config_loss['iters'] = config.quant.weight.optimization.iters
    else:
        config_loss = getattr(config.quant, opt_target).optimization.loss
        config_loss['iters'] = getattr(config.quant, opt_target).optimization.iters
    config_loss['iters'] = config_loss['iters']*0.9  # INFO: anneal to minimum value with 0.7 iters
    config_loss['use_reconstruction_loss'] = 'delta' in param_types
    config_loss['use_round_loss'] = 'alpha' in param_types
    loss_func = LossFunction(layer,**config_loss)

    # move to gpu device
    sample_idxs = torch.randint(low=0,high=cached_inps.shape[0],size=(iters,batch_size))
    torch.set_grad_enabled(True)
    for i in range(iters):
        # import time
        # t0 = time.time()
        # idx = torch.randperm(cached_inps.size(0))[:batch_size]
        idx = sample_idxs[i,:]
        cur_inp = cached_inps[idx]
        cur_out = cached_outs[idx]
        cur_grad = cached_grads[idx] if use_grad else None

        # t1 = time.time()
        # logger.info('data move time {}'.format(t1 - t0))
        optimizer.zero_grad()
        cur_inp.requires_grad_()
        out_quant = layer(cur_inp)

        # t2 = time.time()
        # logger.info('infer time {}'.format(t2 - t1))
        err = loss_func(out_quant, cur_out, cur_grad)
        # t3 = time.time()
        # logger.info('loss time {}'.format(t3 - t2))
        # check nan
        if torch.isnan(err):
            import ipdb; ipdb.set_trace()
        err.backward()  # DEBUG_ONLY: cancel retrain_graph
        # err.backward(retain_graph=True)
        # t4  = time.time()
        # logger.info('backward time {}'.format(t4 - t3))

        # if multi_gpu:
            # raise NotImplementedError
            # for p in opt_params:
            #     link.allreduce(p.grad)

        optimizer.step()
        if scheduler:
            scheduler.step()
        torch.cuda.empty_cache()

    torch.cuda.empty_cache()
    torch.set_grad_enabled(False)

    # Finish optimization, use hard rounding.
    layer.weight_quantizer.soft_targets = False
    if layer.split != 0:
        layer.weight_quantizer_0.soft_targets = False

    return None

