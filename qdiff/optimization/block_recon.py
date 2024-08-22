import torch
# import linklink as link
import logging
from qdiff.quantizer.base_quantizer import lp_loss
from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_model import QuantModel
from qdiff.models.quant_block import BaseQuantBlock
from qdiff.quantizer.base_quantizer import StraightThrough
# from qdiff.quantizer.base_quantizer import AdaRoundQuantizer
from qdiff.utils import save_grad_data, save_in_out_data, LossFunction

logger = logging.getLogger(__name__)

def mv_to_gpu(l_x, device='cuda'):
    if l_x is None:
        pass
    elif isinstance(l_x, list):
        new_l_x = []
        for x in l_x:
            if x is None:
                new_l_x.append(x)
            else:
                new_l_x.append(x.to(device))
        l_x = new_l_x
    elif isinstance(l_x, torch.Tensor):
        l_x = l_x.to(device)
    else:
        import ipdb; ipdb.set_trace()
    return l_x

def block_reconstruction(model: QuantModel, block: BaseQuantBlock, calib_data: torch.Tensor, config, param_types, opt_target):
                         # batch_size: int = 32, iters: int = 20000, weight: float = 0.01, opt_mode: str = 'mse',
                         # asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         # warmup: float = 0.0, act_quant: bool = False, lr: float = 4e-5, p: float = 2.0,
                         # multi_gpu: bool = False, cond: bool = False, is_sm: bool = False):
    """
    Block reconstruction to optimize the output from each block.

    :param model: QuantModel
    :param block: BaseQuantBlock that needs to be optimized
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

    if len(calib_data)==4:
        assert config.model.model_type == 'sdxl'
        cached_inps, cached_outs = save_in_out_data(model, block, calib_data, config, model_type='sdxl')
    else:
        assert config.model.model_type == 'sd'
        cached_inps, cached_outs = save_in_out_data(model, block, calib_data, config, model_type='sd')
    cached_inps = mv_to_gpu(cached_inps, device=device)
    cached_outs = mv_to_gpu(cached_outs, device=device)

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
        cached_grads = save_grad_data(model, block, calib_data, act_quant=False, batch_size=batch_size)  # TODO: reduce act_quant
        cached_grads = cached_grads.to(device)

    # INFO: set the quant states, set_quant_state in SaveData
    # model_quant_weight, model_quant_act = model.get_quant_state()
    # block_quant_weight, block_quant_act = block.get_quant_state()
    # model.set_quant_state(False, False)

    # INFO: setup quant_params and optimizer, use independent lr for each param group
    # DEBUG: currently block_recon only support non-softmax quant_param opt
    opt_params = []  # the param group
    param_group_names = []
    if opt_target == 'weight_and_activation':
        # INFO: should have both of the param groups
        for param_type in param_types['weight']:
            name_ = f"weight.{param_type}"
            param_group_names.append(name_)
            params_ = []
            # INFO: iter through all block modules to get all weight_quantizers
            for layer_name, layer_ in block.named_modules():
                if isinstance(layer_, QuantLayer):
                    params_ += [getattr(layer_.weight_quantizer, param_type)]
                    if layer_.split != 0:
                        params_ += [getattr(layer_.weight_quantizer_0, param_type)]
            opt_params += [{
                'params': params_,
                'lr': getattr(config.quant.weight.optimization.params, param_type).lr,
                }]
        for param_type in param_types['activation']:
            # INFO: iter through all block modules to get all weight_quantizers
            name_ = f"activation.{param_type}"
            param_group_names.append(name_)
            params_ = []
            for layer_name, layer_ in block.named_modules():
                if isinstance(layer_, QuantLayer):
                    params_ = [getattr(layer_.act_quantizer, param_type)]
                    if layer_.split != 0:
                        params_ = [getattr(layer_.act_quantizer_0, param_type)]
            # INFO: a few other layers
            opt_params += [{
                    'params': params_,
                    'lr': getattr(config.quant.activation.optimization.params, param_type).lr,
                    }]

    elif opt_target in ['weight','activation']:
        for param_type in param_types:
            if opt_target == 'weight':
                name_ = f"weight.{param_type}"
                param_group_names.append(name_)
                params_ = []
                # INFO: iter through all block modules to get all weight_quantizers
                for layer_name, layer_ in block.named_modules():
                    if isinstance(layer_, QuantLayer):
                        params_ += [getattr(layer_.weight_quantizer, param_type)]
                        if layer_.split != 0:
                            params_ += [getattr(layer_.weight_quantizer_0, param_type)]
                opt_params += [{
                    'params': params_,
                    'lr': getattr(config.quant.weight.optimization.params, param_type).lr,
                    }]
            elif opt_target == 'activation':
                # INFO: iter through all block modules to get all weight_quantizers
                name_ = f"activation.{param_type}"
                param_group_names.append(name_)
                params_ = []
                for layer_name, layer_ in block.named_modules():
                    if isinstance(layer_, QuantLayer):
                        params_ = [getattr(layer_.act_quantizer, param_type)]
                        if layer_.split != 0:
                            params_ = [getattr(layer_.act_quantizer_0, param_type)]
                # INFO: a few other layers
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
    config_loss['module_type'] = 'block'
    config_loss['use_reconstruction_loss'] = 'delta' in param_types
    config_loss['use_round_loss'] = 'alpha' in param_types
    loss_func = LossFunction(block, **config_loss)

    # move to gpu device
    # sample_idxs = torch.randint(low=0,high=cached_inps.shape[0],size=(iters,batch_size))
    sample_idxs = torch.randint(low=0,high=cached_inps[0].shape[0],size=(iters,batch_size), device=cached_inps[0].device)
    for i in range(iters):
        # import time
        # t0 = time.time()
        # idx = torch.randperm(cached_inps.size(0))[:batch_size]
        idx = sample_idxs[i,:]
        if isinstance(cached_inps, list):
            # 这个对应多输入
            if len(cached_inps)==2:
                # idx = torch.randperm(cached_inps[0].size(0))[:batch_size]
                cur_x = cached_inps[0][idx]
                cur_t = cached_inps[1][idx]
                cur_inp = (cur_x, cur_t)
            else:
                # 针对 QuantTransformerBlock
                cur_inp = []
                # idx = torch.randperm(cached_inps[0].size(0))[:batch_size]
                for j in range(len(cached_inps)):
                    if cached_inps[j] == None:
                        cur_inp.append(None)
                    else:
                        cur_inp.append(cached_inps[j][idx])
                cur_inp = tuple(cur_inp)
        else:
            # idx = torch.randperm(cached_inps.size(0))[:batch_size]  # 随机取样
            cur_inp = cached_inps[idx]
        cur_out = cached_outs[idx]
        cur_grad = cached_grads[idx] if use_grad else None

        optimizer.zero_grad()
        if isinstance(cur_inp, tuple):
            if len(cur_inp) > 2:
                out_quant = block(cur_inp)  # 目前只针对 QuantTransformerblock，该block有多个输入，这时的输入为元组，包含了原本的所有输入
            else:
                out_quant = block(cur_inp[0], cur_inp[1])
        else:
            out_quant = block(cur_inp)

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

    # Finish optimization, use hard rounding.
    # DEBUG: should not always use
    # layer.weight_quantizer.soft_targets = False
    # if layer.split != 0:
        # layer.weight_quantizer_0.soft_targets = False

    return None




    if not include_act_func:
        org_act_func = block.activation_function
        block.activation_function = StraightThrough()

    if not act_quant:
        pass
        # Replace weight quantizer to AdaRoundQuantizer
        # for name, module in block.named_modules():
            # if isinstance(module, QuantLayer):
                # if module.split != 0:
                        # module.weight_quantizer = WeightQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                                # weight_tensor=module.org_weight.data[:, :module.split, ...])
                        # module.weight_quantizer_0 = AdaRoundQuantizer(uaq=module.weight_quantizer_0, round_mode=round_mode,
                                                                # weight_tensor=module.org_weight.data[:, module.split:, ...])
                # else:
                    # module.weight_quantizer = AdaRoundQuantizer(uaq=module.weight_quantizer, round_mode=round_mode,
                                                            # weight_tensor=module.org_weight.data)
                # module.weight_quantizer.soft_targets = True
                # if module.split != 0:
                    # module.weight_quantizer_0.soft_targets = True

        # Set up optimizer
        opt_params = []
        for name, module in block.named_modules():
            if isinstance(module, QuantLayer):
                opt_params += [module.weight_quantizer.alpha]
                if module.split != 0:
                    opt_params += [module.weight_quantizer_0.alpha]
        # optimizer = torch.optim.Adam(opt_params, lr=0.)
        optimizer = torch.optim.Adam(opt_params, lr=1.e-2)
        scheduler = None  # optimizer
    else:
        # Use UniformAffineQuantizer to learn delta
        if hasattr(block.act_quantizer, 'delta') and block.act_quantizer.delta is not None:
            opt_params = [block.act_quantizer.delta]
        else:
            opt_params = []
        
        if hasattr(block, 'attn1'):
            opt_params += [
                block.attn1.act_quantizer_q.delta,
                block.attn1.act_quantizer_k.delta,
                block.attn1.act_quantizer_v.delta,
                block.attn2.act_quantizer_q.delta,
                block.attn2.act_quantizer_k.delta,
                block.attn2.act_quantizer_v.delta]
            if block.attn1.act_quantizer_w.n_bits != 16:
                opt_params += [block.attn1.act_quantizer_w.delta]
            if block.attn2.act_quantizer_w.n_bits != 16:
                opt_params += [block.attn2.act_quantizer_w.delta]
        if hasattr(block, 'act_quantizer_q'):
            opt_params += [
                block.act_quantizer_q.delta,
                block.act_quantizer_k.delta]
        if hasattr(block, 'act_quantizer_w'):
            opt_params += [block.act_quantizer_v.delta]
            if block.act_quantizer_w.n_bits != 16:
                opt_params += [block.act_quantizer_w.delta]

        for name, module in block.named_modules():
            if isinstance(module, QuantLayer):
                if module.act_quantizer.delta is not None:
                    opt_params += [module.act_quantizer.delta]
                if module.split != 0 and module.act_quantizer_0.delta is not None:
                    opt_params += [module.act_quantizer_0.delta]

        optimizer = torch.optim.Adam(opt_params, lr=0.)
        # optimizer = torch.optim.Adam(opt_params, lr=1.e-2)
        scheduler = None
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=0.)

    loss_mode = 'none' if act_quant else 'relaxation'
    rec_loss = opt_mode

    # DEUBG_ONLY: make linear decay faster
    loss_func = LossFunction(block, round_loss=loss_mode, weight=weight, max_count=iters*0.5, rec_loss=rec_loss,
                             b_range=b_range, decay_start=0, warmup=warmup, p=p)

    # Save data before optimizing the rounding
    # cached_inps, cached_outs = save_in_out_data(
        # model, block, calib_data, asym, act_quant, batch_size, keep_gpu=False, cond=cond, is_sm=is_sm)
   # cached_inputs list of [2]: shape: [1024,320,64,64]  [1024,1280]
    sample_idxs = torch.randint(low=0,high=cached_inps[0].shape[0],size=(iters,batch_size), device=cached_inps[0].device)
    for i in range(iters):
        idx = sample_idxs[i,:]
        if isinstance(cached_inps, list):
            # 这个对应多输入
            if len(cached_inps)==2:
                # idx = torch.randperm(cached_inps[0].size(0))[:batch_size]
                cur_x = cached_inps[0][idx]
                cur_t = cached_inps[1][idx]
                cur_inp = (cur_x, cur_t)
            else:
                # 针对 QuantTransformerBlock
                cur_inp = []
                # idx = torch.randperm(cached_inps[0].size(0))[:batch_size]
                for j in range(len(cached_inps)):
                    if cached_inps[j] == None:
                        cur_inp.append(None)
                    else:
                        cur_inp.append(cached_inps[j][idx])
                cur_inp = tuple(cur_inp)
        else:
            # idx = torch.randperm(cached_inps.size(0))[:batch_size]  # 随机取样
            cur_inp = cached_inps[idx]
        cur_out = cached_outs[idx]
        cur_grad = cached_grads[idx] if opt_mode != 'mse' else None

        optimizer.zero_grad()
        if isinstance(cur_inp, tuple):
            if len(cur_inp) > 2:
                out_quant = block(cur_inp)  # 目前只针对 QuantTransformerblock，该block有多个输入，这时的输入为元组，包含了原本的所有输入
            else:
                out_quant = block(cur_inp[0], cur_inp[1])
        else:
            out_quant = block(cur_inp)

        err = loss_func(out_quant, cur_out, cur_grad)
        if torch.isnan(err):
            import ipdb; ipdb.set_trace()
        # err.backward(retain_graph=True)
        err.backward()   # DEBUG_ONLY: cancel the retain_graph
        if multi_gpu:
            raise NotImplementedError
        #     for p in opt_params:
        #         link.allreduce(p.grad)
        optimizer.step()
        if scheduler:
            scheduler.step()

    torch.cuda.empty_cache()

    # Finish optimization, use hard rounding.
    for name, module in block.named_modules():
        if isinstance(module, QuantLayer):
            module.weight_quantizer.soft_targets = False
            if module.split != 0:
                module.weight_quantizer_0.soft_targets = False

    # Reset original activation function
    if not include_act_func:
        block.activation_function = org_act_func

