import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import time # DEBUG_ONLY
import math

logger = logging.getLogger(__name__)

# ---------- the quantizers ------------------
class BaseQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param per_group: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, quant_config):
        super(BaseQuantizer, self).__init__()
        # --> Unpack the quant configurations
        # --> the MUST configs
        self.n_bits = quant_config.n_bits
        self.mixed_precision = quant_config.get('mixed_precision')
        self.timestep_wise = quant_config.get("timestep_wise")
        if self.mixed_precision is not None:
            # generate idx for list of quant parameters
            self.bit_idx = self.mixed_precision.index(self.n_bits)
        else:
            self.bit_idx = 0
        # INFO: support timestep-wise quant_params for act
        self.cur_timestep_id = 0  # assigned outside, in solver, should be [0~1000]

        self.per_group = quant_config.per_group
        self.channel_dim = quant_config.get("channel_dim", 0)
        self.scale_method = quant_config.scale_method
        self.round_mode = quant_config.round_mode
        # --> the optional configs
        self.sym = quant_config.get('sym', False)
        self.running_stat = quant_config.get('running_stat',False)  # INFO: use running_stat to accmulate quant_params, used in activation
        self.momentum = 0.95 if self.running_stat else None
        # INFO: seemingly always_zero means x_min should be 0, used in softmax quant, maybe rermove later, or a better name?
        self.always_zero = quant_config.get('always_zero', False)

        # quant parameters: [bitwidth, timestep]
        if self.mixed_precision is not None:
            self.n_bitwidth = len(self.mixed_precision)
        else:
            self.n_bitwidth = 1
        if self.timestep_wise:
            self.n_timestep = 1000  # the complete space
        else:
            self.n_timestep = 1

        # self.register_buffer('delta_list', torch.zeros([n_bitwidth, n_timestep]).fill_(-1.))
        # self.register_buffer('zero_point_list', torch.zeros([n_bitwidth, n_timestep]).fill_(-1.))
        self.register_buffer('delta_list', None)
        self.register_buffer('zero_point_list', None)
        self.register_buffer('delta', None)
        self.register_buffer('zero_point', None)

        # quant specs
        self.init_done = False

        # ---- attr for rounding ----
        self.register_buffer('alpha', None)
        self.soft_targets = True
        if self.round_mode == 'learned_hard_sigmoid':
            # params for sigmoid function
            self.gamma, self.zeta = -0.1, 1.1
            self.beta = 2/3

    def rounding(self, x: torch.Tensor):
        '''The Rounding Function: use delta & zero_point, get x_quant
        '''
        if self.round_mode == 'nearest':
            x_int = torch.round(x / self.delta)
        elif self.round_mode == 'nearest_ste':
            x_int = round_ste(x / self.delta)
        elif self.round_mode == 'stochastic':
            x_floor = torch.floor(x / self.delta)
            rest = (x / self.delta) - x_floor  # rest of rounding
            x_int = x_floor + torch.bernoulli(rest)
            logger.info('Draw stochastic sample')
        elif self.round_mode == 'learned_hard_sigmoid':
            x_floor = torch.floor(x / self.delta)
            if self.soft_targets:
                soft_targets = torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
                x_int = x_floor + soft_targets
                if x_floor.dtype == torch.float16:
                    x_int = x_int.to(torch.float16)
            else:
                x_int = x_floor + (self.alpha >= 0).float()
                if x_floor.dtype == torch.float16:
                    x_int = x_int.to(torch.float16)
        else:
            raise ValueError('Wrong rounding mode')

        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        return x_quant

    def get_soft_targets(self):
        soft_targets = torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)
        return soft_targets

    def forward(self, x: torch.Tensor):

        if self.init_done is not True:
            if self.mixed_precision is not None:
                # save the n_bits, bit_idx to avoid overwrite
                for i_bitwidth, n_bits in enumerate(self.mixed_precision):
                    assert 2 <= n_bits <= 16, 'bitwidth not supported'
                    self.init_quant_params(x, self.per_group, momentum=self.running_stat, n_bits=n_bits)
                    # logging.warning('forwarding qnn without initialize quant params, the data is used for init')
            else:
                self.init_quant_params(x, self.per_group, momentum=self.running_stat)

        # always indexing from the delta_list and zp_list
        if self.timestep_wise:
            self.delta = self.delta_list[self.bit_idx, self.cur_timestep_id]
            self.zero_point = self.zero_point_list[self.bit_idx, self.cur_timestep_id]
        else:
            self.delta = self.delta_list[self.bit_idx, 0]
            self.zero_point = self.zero_point_list[self.bit_idx, 0]

        assert not torch.all(self.delta == -1) # check if not -1

        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        # start quantization
        # print(f"x shape {x.shape} delta shape {self.delta.shape} zero shape {self.zero_point.shape}")
        try:
            x_int = round_ste(x / self.delta) + self.zero_point
        except:
            import ipdb; ipdb.set_trace()

        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        # import ipdb; ipdb.set_trace()
        x_quant_ = self.rounding(x)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

    def init_quant_params(self, x: torch.Tensor, per_group: bool = False, momentum=False, n_bits=None):
        '''
        - could take different input shapes(both W and A): \
                    - [B,C,H,W] (acitvation)
                    - [C1,C2,K,K] (conv weight)
                    - [C1,C2] (linear weight)
        - support different init scale: 'min-max' and 'min_lq_loss'
        '''
        # default use self.n_bits, specified for mixed_precision
        if self.mixed_precision is not None:
            i_bitwidth = self.mixed_precision.index(n_bits)
        else:
            i_bitwidth = 0
        if n_bits is None:
            n_bits = self.n_bits
        n_levels = 2 ** n_bits if not self.sym else 2 ** (n_bits - 1) - 1

        delta, zero_point = None, None
        x_shape = x.shape
        if isinstance(x, nn.Parameter):  # some input weight is parameter, process the tensor
            x = x.data.clone()
        # the min-max quantization parameter init
        if per_group: # apply channel-wise scaling
            if per_group == 'channel':
                if self.channel_dim == 0:
                    n_channel = x.shape[0]  # n_groups = n_channel
                    x = x.reshape([n_channel,-1])
                elif self.channel_dim == 1:
                    n_channel = x.shape[1]
                    x.permute(1, 0, *range(2, x.dim()))
                    x = x.reshape([n_channel,-1])
            elif per_group == 'token':
                assert isinstance(self,ActQuantizer)
                try:
                    assert len(x.shape) == 3
                except:
                    import ipdb; ipdb.set_trace()
                # print(self.module_name, x.shape) # shape: [BS, n_token, C_in]
                n_token = x.shape[1]
                x = x.permute([1,0,2]).reshape([n_token,-1])  # [n_token, BS*C_in]
            else:
                raise NotImplementedError
        else: # apply tensor-wise scaling, return a singular value
            x = x.reshape(-1)

        x_min = x.min(dim=-1)[0]
        x_min[x_min>0] = 0.
        x_max = x.max(dim=-1)[0] # INFO: used for some meaningless range
        x_max[x_max<0] = 0.

        if self.momentum:
            if not hasattr(self,'x_min'):
                # for the 1st time, save as x_min & x_max
                assert not hasattr(self, 'x_max') # both haven't initialized
                self.x_min = x_min
                self.x_max = x_max
            else:
                # momentum update of the x_min & x_max (from 2nd iteration)
                self.x_min = self.x_min*self.momentum + x_min*(1-self.momentum)
                self.x_max = self.x_max*self.momentum + x_max*(1-self.momentum)
                x_min = self.x_min
                x_max = self.x_max

        # DIRTY: fix some legacy checkpoints, where 'min_max' is 'max'
        if self.scale_method == 'max':
            self.scale_method = 'min_max'

        if self.scale_method == 'min_max':
            x_absmax = torch.maximum(x_min.abs(),x_max.abs())
            # x_absmax = torch.maximum(x.min(dim=-1)[0].abs(),x.max(dim=-1)[0].abs())  # ???: donnot know why not using x_min
            if self.sym: # symmetric_quant
                delta = x_absmax/n_levels
            else:
                delta = (x_max - x_min)/(n_levels-1)
            eps = 1.e-6
            if delta.min() < eps:  # if contain small range
                delta = delta.fill_(eps)
                warnings.warn('For layer "{}", quant stept size close to zero, set as EPS:{}'.format(self.module_name, eps))

            if self.always_zero or self.sym: # always set zero_point as 0, no clue what it is
                zero_point = torch.zeros_like(delta, device=delta.device)
            else:
                zero_point = torch.round(-x_min/delta)  # ???: why -x_min, save as unsigned int

        elif self.scale_method == 'grid_search_lp':

            eps=1.e-5
            best_score = 1.e10
            step_size = 0.01
            range_scaling = torch.arange(0,1,step_size).to(x.device)
            n_step = len(range_scaling)
            scaled_max = x_max.unsqueeze(0)*range_scaling.unsqueeze(-1)
            scaled_min = x_min.unsqueeze(0)*range_scaling.unsqueeze(-1)
            x_ranged = x.unsqueeze(0).repeat(tuple([n_step]+[1]*len(x.shape))) # [N_bs, x_shape]
            x_q = self.quantize(x_ranged,scaled_max,scaled_min,n_batch=n_step)
            lp_loss_ranged = lp_loss(x_ranged, x_q, p=2., reduction='none',n_batch=n_step)  # [N_bs]
            min_idx = torch.argmin(lp_loss_ranged) # the min idx 
            if not self.always_zero:
                delta = (scaled_max[min_idx]-scaled_min[min_idx])/(2**n_bits-1)
                # DEBUG: delta seems to could be all-zero, is it right though?
                zero_point = (-scaled_min[min_idx]/delta+eps).round()
            else:
                delta = scaled_max[min_idx]/(2**n_bits-1)
                zero_point = torch.zeros_like(delta)
        else:
            # import ipdb; ipdb.set_trace()
            raise NotImplementedError

        # init the rounding parameters
        if self.round_mode == 'learned_hard_sigmoid':
            # logger.info('Init alpha to be FP32')
            delta_shape = [1]*len(x.shape)  # temporarily reshape self.delta to fit current x shape, x is reshaped into 2-dim, note that the x_shape are longer than x.shape
            delta_shape[:len(delta.shape)] = delta.shape
            delta_ = delta.reshape(delta_shape)
            x_floor = torch.floor(x / delta_)
            rest = (x / delta_) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            # reshape alpha to fit original x_shape, should be the same
            self.alpha = alpha.reshape(x_shape)

        # INFO: align the shape of the delta & zero_point
        # whether delta is single-value (tensor-wise) or of shape [C] (channel-wise)
        assert delta.shape == zero_point.shape
        if per_group == 'token':
            delta = delta.reshape([1,n_token, 1])  # fit the original x_shape
            zero_point = zero_point.reshape([1,n_token, 1])  # fit the original x_shape
        else:
            shape_ = [1]*len(x_shape)  # note that the x.shape is not same with x_shape
            shape_[:len(delta.shape)] = delta.shape
            delta = delta.reshape(shape_)
            assert isinstance(delta, torch.Tensor), "during init, delta should be a tensor, instead of type: {}".format(type(self.delta))
            zero_point = zero_point.reshape(shape_)
            if self.channel_dim == 1:
                self.delta = self.delta.permute(1, 0, *range(2, x.dim()))
                self.zero_point = self.zero_point.permute(1, 0, *range(2, x.dim()))

        # INFO: asssign delta and zp to self.delta/zp
        # DIRTY: since delta could be [C], [C,1], [C,1,1,1], the len could not be determined when init delta_list (by x)
        if not self.timestep_wise:
            assert self.cur_timestep_id == 0 # DBEUG_ONLY
        if self.delta_list is None: # init the delta_list according to actual delta/zp shape
            self.delta_list = torch.zeros([self.n_bitwidth, self.n_timestep]+list(delta.shape), dtype=delta.dtype).fill_(-1.).to(x.device)
            self.zero_point_list = torch.zeros([self.n_bitwidth, self.n_timestep]+list(zero_point.shape), dtype=zero_point.dtype).fill_(-1.).to(x.device)
        self.delta_list[i_bitwidth, self.cur_timestep_id] = delta  # when timestep_wise=False, self.cur_timestep_id=-1, index the only one
        self.zero_point_list[i_bitwidth, self.cur_timestep_id] = zero_point

    def quantize(self, x, x_max, x_min, n_batch=None):
        '''quantizing with given x_max, x_min, instead using delta and zero_point'''
        # when max, min has shape [N_bs], x has [N_bs, x_shape]
        # x would have multiple possible shapes
        if n_batch is not None:
            assert x_max.shape[0] == n_batch
            assert x_min.shape[0] == n_batch
            assert x.shape[0] == n_batch

        # quantize with given max and min values
        eps=1.e-4
        delta = (x_max - x_min) / (2 ** self.n_bits - 1) if not self.always_zero else x_max / (2 ** self.n_bits - 1)
        zero_point = (- x_min / (delta + eps)).round() if not self.always_zero else 0
        if n_batch is not None:
            delta = delta.reshape(list(delta.shape)+[1]*(len(x.shape) - len(delta.shape)))
            zero_point = zero_point.reshape(list(zero_point.shape)+[1]*(len(x.shape) - len(zero_point.shape)))
        # we assume weight quantization is always signed
        x_int = torch.round(x / (delta + eps))
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_dequant = (x_quant - zero_point) * delta

        # check nan
        if torch.isnan(x_dequant).any() > 0:
            raise ValueError('nan exist in x_q')
        return x_dequant


    def bitwidth_refactor(self, refactored_bit: int):
        assert 2 <= refactored_bit <= 16, 'bitwidth not supported'
        self.n_bits = refactored_bit
        if self.mixed_precision is not None:
            # self.bit_idx = int(math.log2(self.n_bits))-1 # only used for mixed precision
            # i_bitwidth = self.mixed_precision.index(self.n_bits)
            self.bit_idx = self.mixed_precision.index(self.n_bits)


    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, per_group={per_group}, round_mode={round_mode}'
        return s.format(**self.__dict__)

class WeightQuantizer(BaseQuantizer):
    """
    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param per_group: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, quant_config):
        super(WeightQuantizer, self).__init__(quant_config)

class ActQuantizer(BaseQuantizer):
    """
    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param per_group: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, quant_config):
        super(ActQuantizer, self).__init__(quant_config)


# ---------- some quantizer util funcs ------------------
class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x

def lp_loss(pred, tgt, p=2, reduction='none', n_batch=None):
    """
    loss function measured in L_p Norm
    """
    # INFO: add support for n_batch
    # reduce 'none means restricted to L2 norm, and reduce the dimension (more sum)
    # the input x could be [A,B],[A,B,C],[A,B,C,D], when has batch [N_bs,A,B]
    # ---------------------------------------------------------------------------
    # INFO: the original lp_loss func only sums on the 2nd dim, which is confusing
    # modify to sum all elements
    assert pred.shape == tgt.shape
    reduce_dims_except_0 = tuple(range(1,len(pred.shape)))
    reduce_dims_except_1 = tuple(range(2,len(pred.shape)))
    if n_batch is not None:
        assert pred.shape[0] == n_batch
        assert tgt.shape[0] == n_batch
        if reduction == 'none':
            # INFO: if tensor-wsie & none reduction, reduce_dim_except_1 is ()
            if len(reduce_dims_except_1) == 0:
                return ((pred-tgt).abs()**2).sum(dim=1)
            else:
                return ((pred-tgt).abs()**2).sum(dim=reduce_dims_except_1).mean(dim=1)
        elif reduction == 'all':
            return (pred-tgt).abs().pow(p).mean(dim=reduce_dims_except_0)
        else:
            raise NotImplementedError
    else:
        if reduction == 'none':
            return ((pred-tgt).abs()**2).sum(dim=reduce_dims_except_0).mean()
        elif reduction == 'all':
            return (pred-tgt).abs().pow(p).mean()
        else:
            raise NotImplementedError
