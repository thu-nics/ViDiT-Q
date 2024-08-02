import torch.nn as nn
import torch
import logging
from typing import Union, Optional, Dict, Any, Tuple

from qdiff.models.quant_layer import QuantLayer
from qdiff.models.stdit_quant_layer import QuantSpatialAttnLinear, QuantTemporalAttnLinear, QuantCrossAttnLinear
from qdiff.models.dit_quant_layer import QuantAttnLinearImg, QuantCrossAttnLinearImg
from qdiff.models.quant_block import BaseQuantBlock, TransformerBlock, QuantTransformerBlock, get_specials
from qdiff.quantizer.base_quantizer import StraightThrough, BaseQuantizer, WeightQuantizer, ActQuantizer

logger = logging.getLogger(__name__)

def pattern_in(text, pattern):
    patterns = pattern.split(".")
    texts = text.split(".")
    for i in range(len(texts)):
        for j in range(len(patterns)):
            if patterns[j] == "*":
                continue
            elif "[" in patterns[j] and "]" in patterns[j]:
                tmp_pattern = patterns[j][1:-1].split("-")
                int_range = list(range(int(tmp_pattern[0]), int(tmp_pattern[1]) + 1))
                str_range = [str(x) for x in int_range]
                if texts[i + j] in str_range:
                    continue
                else:
                    break
            else:
                if texts[i + j] == patterns[j]:
                    continue
                else:
                    break
        else:
            return True
    return False

class QuantModel(nn.Module):
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, model_type="opensora", **kwargs):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        # self.dtype = torch.float32

        self.weight_quant = False if weight_quant_params is None else True
        self.act_quant = False if act_quant_params is None else True
        self.model_type = model_type
        self.timestep_wise = act_quant_params.get('timestep_wise', False)
        self.specials = get_specials(model_type)

        self.model = model
        self.in_channels = model.in_channels
        if hasattr(model, 'image_size'):
            self.image_size = model.image_size
        # self.specials = get_specials()  # some nn.Modules require special process
        logger.info(f"\n --------------- refactoring quant layers --------------- \n")
        self.quant_layer_refactor(self.model, weight_quant_params, act_quant_params)
        # logger.info(f"\n --------------- refactoring quant blocks --------------- \n")
        # self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)
        # self.set_module_name_for_quantizer(module=self.model)  # add the module name as attribute for each quantizer
        self.quant_params_dict = {}  # init the quant_params_dict as empty


    def quant_layer_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, prefix=""):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantLayer
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        prev_quantmodule = None
        for name, child_module in module.named_children():
            full_name = prefix+name if prefix else name
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                tmp_module = child_module
                # INFO: for stdit model, assign quant_{spatial/temporal/cross}_attn_layers respectively
                if self.model_type == 'opensora':
                    assert isinstance(tmp_module, nn.Linear)  # only linear layers to quantizr in stdit model
                if '.attn.' in full_name:
                    if self.model_type == 'opensora':
                        setattr(module, name, QuantSpatialAttnLinear(\
                            child_module, weight_quant_params, act_quant_params))
                    elif self.model_type == 'pixart':
                        setattr(module, name, QuantAttnLinearImg(\
                            child_module, weight_quant_params, act_quant_params))
                elif 'cross_attn' in full_name:
                    if self.model_type == 'opensora':
                        setattr(module, name, QuantCrossAttnLinear(\
                            child_module, weight_quant_params, act_quant_params))
                    elif self.model_type == 'pixart':
                        setattr(module, name, QuantCrossAttnLinearImg(\
                            child_module, weight_quant_params, act_quant_params))
                elif 'attn_temp' in full_name:
                    setattr(module, name, QuantTemporalAttnLinear(\
                        child_module, weight_quant_params, act_quant_params))
                else:
                    setattr(module, name, QuantLayer(
                        child_module, weight_quant_params, act_quant_params))
                prev_quantmodule = getattr(module, name)
                # logger.info(f"\n origional module: {name}:{tmp_module}, \n new module {prev_quantmodule}")
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_layer_refactor(child_module, weight_quant_params, act_quant_params, prefix=full_name+'.')  # recursive call


    # def quant_block_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        # for name, child_module in module.named_children():
            # if type(child_module) in self.specials:
                # tmp_module = child_module
                # # if self.specials[type(child_module)] in [QuantBasicTransformerBlock, QuantAttention, QuantTransformerBlock]:
                # if self.specials[type(child_module)] in [QuantTransformerBlock]:
                    # setattr(module, name, self.specials[type(child_module)](child_module, act_quant_params))
                    # # logger.info(f"\n origional block: {name}:{tmp_module}, \n new block {module}")
                # # elif self.specials[type(child_module)] == QuantSMVMatMul:
                    # # setattr(module, name, self.specials[type(child_module)](
                        # # act_quant_params, sm_abit=self.sm_abit))
                    # # logger.info(f"\n origional block: {name}:{tmp_module}, \n new block {module}")
                # # elif self.specials[type(child_module)] == QuantQKMatMul:
                    # # setattr(module, name, self.specials[type(child_module)](
                        # # act_quant_params))
                    # # logger.info(f"\n origional block: {name}:{tmp_module}, \n new block {module}")
                # else:
                    # tmp_module = child_module
                    # setattr(module, name, self.specials[type(child_module)](child_module, act_quant_params))
                    # # logger.info(f"\n origional block: {name}:{tmp_module}, \n new block {module}")
            # else:
                # self.quant_block_refactor(child_module, weight_quant_params, act_quant_params)


    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # update the QuantModel quant_state
        self.weight_quant = weight_quant
        self.act_quant = act_quant

        for m in self.model.modules():
            if isinstance(m, (QuantLayer, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

        # Keep the fp_layer_list as FP
        if hasattr(self, "fp_layer_list"):
            self.set_layer_quant(model=self, module_name_list=self.fp_layer_list, quant_level='per_layer', weight_quant=False, act_quant=False, prefix="")

    def get_quant_state(self):
        return self.weight_quant, self.act_quant


    def set_module_name_for_quantizer(self, module, prefix=""):
        '''set the nn.Module name for each quantizer'''
        for name_, module_ in module.named_children():
            full_name = prefix + name_ if prefix else name_
            torch.cuda.empty_cache()
            if isinstance(module_, BaseQuantizer):  # end with quantizer module
                setattr(module_,'module_name',full_name)
                # print(module_, full_name)
            else:
                self.set_module_name_for_quantizer(module=module_, prefix=full_name+'.')

    def set_timestep_for_quantizer(self, t, module=None):
        if module is None:
            module = self
        '''set the nn.Module name for each quantizer'''
        for name_, module_ in module.named_children():
            if isinstance(module_, BaseQuantizer):  # end with quantizer module
                setattr(module_,'cur_timestep_id',t)
            else:
                self.set_timestep_for_quantizer(t, module=module_)


    def set_timestep_id_for_quantlayer(self, t, module=None):
        if module is None:
            module = self
        '''set the nn.Module name for each quantizer'''
        for name_, module_ in module.named_children():
            # if isinstance(module_, QuantSpatialAttnLinear):
            #     print("xxxxx")
            if isinstance(module_, QuantLayer):  # end with quantizer module
                # cur_timestep_id = get_cur_timestep_id(t, module_.timerange)
                setattr(module_,'cur_timestep_id', t)

            else:
                self.set_timestep_id_for_quantlayer(t, module=module_)


    def repeat_timestep_wise_quant_params(self, ts, module=None):
        if module is None:
            module = self
        '''set the nn.Module name for each quantizer'''
        for name_, module_ in module.named_children():
            if isinstance(module_, BaseQuantizer):  # end with quantizer module
                if module_.timestep_wise:
                    # Plan 1: repeat the quant params (delta, zp) based on zt
                    choose_idx = torch.unique(ts)
                    assert 1000%len(choose_idx) == 0
                    module_.delta_list = torch.repeat_interleave(module_.delta_list[:,choose_idx,:],1000//len(choose_idx),dim=1)
                    module_.zero_point_list = torch.repeat_interleave(module_.zero_point_list[:,choose_idx,:],1000//len(choose_idx),dim=1)
            else:
                self.repeat_timestep_wise_quant_params(ts, module=module_)



    def set_quant_init_done(self, quantizer_type_name, module=None):
        if module is None:
            module = self.model  # use full model when empty module input
        '''set init_done name for each quantizer'''
        if quantizer_type_name == "weight":
            quantizer_type = WeightQuantizer
        elif quantizer_type_name == "activation":
            quantizer_type = ActQuantizer
        else:
            raise NotImplementedError

        for name_, module_ in module.named_children():
            torch.cuda.empty_cache()
            if isinstance(module_, quantizer_type):  # end with quantizer module
                module_.init_done = True
            else:
                self.set_quant_init_done(quantizer_type_name, module_)


    def get_quant_params_dict(self, module=None, prefix="", dtype=torch.float32):
        # iter through all quantizers, get the buffers
        if module is None:
            module = self.model
            self.quant_params_dict = {}
        quantizer_type = BaseQuantizer
        # recursively iter through all quantizers
        for name, module_ in module.named_children():
            full_name = prefix + name if prefix else name
            torch.cuda.empty_cache()
            if isinstance(module_, quantizer_type):
                # pack the dict into the 'module_name'
                # [buffers_(OrderdDict), parameters(OrderedDict)]
                self.quant_params_dict[module_.module_name] = []
                self.quant_params_dict[module_.module_name].append(module_._buffers)
                self.quant_params_dict[module_.module_name].append(module_._parameters)
            else:
                self.get_quant_params_dict(module=module_, prefix=full_name+'.')

        return self.quant_params_dict


    def set_quant_params_dict(self, quant_params_dict, module=None, load_buffer_only=True, dtype=torch.float32):
        # iter through all quantizers, set the buffers with self.quant_param_dict
        # quant_parma_dict: ['conv_in.weight_quantizer'] is a tuple, 1st is _bufferes, 2nd is _params()]
        # load_buffer_only: when `quantized_inference`, should only load the buffers (the saved ckpt should be all buffers)
        # when resuming quantization, load both the buffers and the parameters
        if module is None:
            module = self.model

        quantizer_type = BaseQuantizer

        # recursively iter through all quantizers
        for name, module_ in module.named_children():
            torch.cuda.empty_cache()
            if isinstance(module_, quantizer_type):
               # unpack the dict
                if load_buffer_only:
                    assert len(quant_params_dict[module_.module_name][1]) == 0  # parameters() has no element
                    for name, quant_params in quant_params_dict[module_.module_name][0].items():  # use module_name to index the dict
                        setattr(module_, name, quant_params.to(dtype) if quant_params is not None else None)
                else:
                    # set buffer
                    for name, quant_params in quant_params_dict[module_.module_name][0].items():
                        setattr(module_, name, quant_params.to(dtype) if quant_params is not None else None)
                    # set parameter
                    for name, quant_params in quant_params_dict[module_.module_name][0].items():
                        setattr(module_, name, quant_params.to(dtype) if quant_params is not None else None)
            else:
                self.set_quant_params_dict(quant_params_dict=quant_params_dict, module=module_)


    def replace_quant_buffer_with_parameter(self, opt_d, module=None):
        if module is None:
            module = self.model

        for opt_target in opt_d.keys():

            if opt_target == 'weight':
                quantizer_type = WeightQuantizer
            elif opt_target == 'activation':
                quantizer_type = ActQuantizer

            for name, module_ in module.named_children():
                # print(module_)
                torch.cuda.empty_cache()
                if isinstance(module_, quantizer_type):
                    if opt_d[opt_target] is not None:
                        for param_type in opt_d[opt_target]:
                            # skip the conversion for layers that remain FP
                            if module_.module_name.split('.')[0] in self.fp_layer_list:
                                continue
                            else:
                                buffer_ = getattr(module_, param_type)
                                assert isinstance(buffer_, torch.Tensor)
                                delattr(module_, param_type)
                                module_.register_parameter(param_type, nn.Parameter(buffer_))
                else:
                    self.replace_quant_buffer_with_parameter(opt_d, module=module_)


    def replace_quant_parameter_with_buffers(self, opt_d, module=None):
        if module is None:
            module = self.model

        quantizer_type = BaseQuantizer

        for opt_target in opt_d.keys():

            if opt_target == 'weight':
                quantizer_type = WeightQuantizer
            elif opt_target == 'activation':
                quantizer_type = ActQuantizer

            for name, module_ in module.named_children():
                torch.cuda.empty_cache()
                if isinstance(module_, quantizer_type):
                    # if opt_d[opt_target] is not None:
                    if opt_d[opt_target] is not None:
                        for param_type in opt_d[opt_target]:
                            if module_.module_name.split('.')[0] in self.fp_layer_list:
                                continue
                            else:
                                buffer_ = getattr(module_, param_type).data
                                assert isinstance(buffer_, torch.Tensor)
                                delattr(module_, param_type)
                                module_.register_buffer(param_type, buffer_)
                    # if opt_d['activation'] is not None:
                    #     for param_type in opt_d['activation']:
                    #         buffer_ = getattr(module_, param_type)
                    #         assert isinstance(buffer_, torch.Tensor)
                    #         delattr(module_, param_type)
                    #         module_.register_buffer(param_type, buffer_)
                else:
                    self.replace_quant_parameter_with_buffers(opt_d, module=module_)
    
    # ---------------- General -----------------
    def forward(self,
                x,
                t,
                y,
                **kwargs,):
        # compatible with diffusers UNetCondition2D forward function
        if self.timestep_wise:
            assert torch.all(t == t[0])
            self.set_timestep_for_quantizer(t[0].item())
            
        self.set_timestep_id_for_quantlayer(t[0].item())  # used for timestep-wise smooth quant alpha


        # DEBUG: force FP for some timesteps
        # the begining of 2 range are used for weight quantization, should not make them FP
        if isinstance(t, torch.Tensor):
            t_ = t[0].item()
        else:
            assert isinstance(t_, float)

        return self.model(x,
                          t,
                          y,
                          **kwargs,)

    # ---------- For Diffuser Models -----------
    # def forward(self,
    #             sample: torch.FloatTensor,
    #             timestep: Union[torch.Tensor, float, int],
    #             encoder_hidden_states: torch.Tensor,
    #             class_labels: Optional[torch.Tensor] = None,
    #             timestep_cond: Optional[torch.Tensor] = None,
    #             attention_mask: Optional[torch.Tensor] = None,
    #             cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    #             added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    #             down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    #             mid_block_additional_residual: Optional[torch.Tensor] = None,
    #             down_intrablock_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
    #             encoder_attention_mask: Optional[torch.Tensor] = None,
    #             return_dict: bool = True,):
    #     # compatible with diffusers UNetCondition2D forward function
    #     return self.model(sample, timestep, encoder_hidden_states, class_labels, timestep_cond, attention_mask, cross_attention_kwargs,
    #                     added_cond_kwargs, down_block_additional_residuals, mid_block_additional_residual, down_block_additional_residuals, encoder_attention_mask,
    #                     return_dict)

    # def set_running_stat(self, running_stat: bool, sm_only=False):
        # for m in self.model.modules():
            # if isinstance(m, (QuantBasicTransformerBlock, QuantTransformerBlock)):
                # if sm_only:
                    # m.attn1.act_quantizer_w.running_stat = running_stat
                    # m.attn2.act_quantizer_w.running_stat = running_stat
                # else:
                    # m.attn1.act_quantizer_q.running_stat = running_stat
                    # m.attn1.act_quantizer_k.running_stat = running_stat
                    # m.attn1.act_quantizer_v.running_stat = running_stat
                    # m.attn1.act_quantizer_w.running_stat = running_stat
                    # m.attn2.act_quantizer_q.running_stat = running_stat
                    # m.attn2.act_quantizer_k.running_stat = running_stat
                    # m.attn2.act_quantizer_v.running_stat = running_stat
                    # m.attn2.act_quantizer_w.running_stat = running_stat
            # if isinstance(m, QuantLayer) and not sm_only:
                # m.set_running_stat(running_stat)


    def set_grad_ckpt(self, grad_ckpt: bool):
        for name, m in self.model.named_modules():
            # elif isinstance(m, QuantResBlock):
                # logger.info(name)
                # m.use_checkpoint = grad_ckpt
            if isinstance(m, (QuantTransformerBlock, TransformerBlock)):
                # logger.info(name)
                m.checkpoint = grad_ckpt

    def set_smooth_quant(self, smooth_quant, smooth_quant_running_stat):
        # update the QuantModel quant_state
        self.smooth_quant_stat = smooth_quant_running_stat
        
        for m in self.model.modules():
            if isinstance(m, (QuantLayer)):
                m.smooth_quant = smooth_quant
                m.smooth_quant_running_stat = smooth_quant_running_stat
                
    def set_layer_smooth_quant(self, model, module_name_list, smooth_quant, smooth_quant_running_stat, prefix=""):
        for name, module in model.named_children():
            full_name = prefix + name if prefix else name
            if isinstance(module, QuantLayer):
                for module_name in module_name_list:
                    if pattern_in(full_name, module_name) or pattern_in(full_name, 'model.'+ module_name):
                    # if module_name in full_name or ('model.'+ module_name) in full_name:
                        module.smooth_quant_running_stat = smooth_quant_running_stat
                        module.smooth_quant = smooth_quant
                        logger.info(f"{full_name}: smooth_quant={smooth_quant} | smooth_quant_running_stat={smooth_quant_running_stat}")
            else:
                self.set_layer_smooth_quant(model=module, module_name_list=module_name_list, smooth_quant=smooth_quant, smooth_quant_running_stat=smooth_quant_running_stat, prefix=full_name+".")


    def set_layer_quant(self, model=None, module_name_list=[], group_list=[], group_ignore=[],  quant_level='per_layer', weight_quant=True, act_quant=False, prefix=""):
        '''
        progressively quantize the groups or layers, which is different from the func in the quant_error.py
        quantify all layers in the module_list or group_list at once
        group_ignore: if quant_level is 'per_group', selectively ignore the quantization of certain groups
        '''
        if quant_level=='per_group':
            for name, module in model.named_children():
                full_name = prefix + name if prefix else name
                if isinstance(module, QuantLayer):
                    for module_class in group_list:
                        if module_class == 'attn':
                            if module_class in full_name and not 'cross_attn' in full_name and not 'attn_temp' in full_name:
                                if all(element not in full_name for element in group_ignore):
                                    module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                                    torch.cuda.empty_cache()
                                    logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
                        else:        
                            if module_class in full_name:
                                if all(element not in full_name for element in group_ignore):
                                    module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                                    torch.cuda.empty_cache()
                                    logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
                else:
                    self.set_layer_quant(model=module, module_name_list=module_name_list, group_list=group_list, group_ignore=group_ignore, quant_level='per_group', weight_quant=weight_quant, act_quant=act_quant, prefix=full_name+".")

        if quant_level=='per_layer':
            for name, module in model.named_children():
                full_name = prefix + name if prefix else name
                if isinstance(module, QuantLayer):
                    for module_name in module_name_list:
                        if pattern_in(full_name, module_name) or pattern_in(full_name, 'model.'+ module_name):
                        # if module_name in full_name or ('model.'+ module_name) in full_name:
                            module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                            torch.cuda.empty_cache()
                            logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
                else:
                    self.set_layer_quant(model=module, module_name_list=module_name_list, group_list=group_list, group_ignore=group_ignore, quant_level='per_layer', weight_quant=weight_quant, act_quant=act_quant, prefix=full_name+".")

        if quant_level=='per_block':
            for name, module in model.named_children():
                full_name = prefix + name if prefix else name
                if isinstance(module, BaseQuantBlock):
                    for module_name in module_name_list:
                        module_name = 'model.'+module_name
                        if module_name == full_name:
                            module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                            torch.cuda.empty_cache()
                            logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
                elif isinstance(module, QuantLayer):
                    for module_name in module_name_list:
                        module_name = 'model.'+module_name
                        if module_name == full_name:
                            module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
                            torch.cuda.empty_cache()
                            logger.info(f"{full_name}: weight_quant={weight_quant}, act_quant={act_quant}")
                else:
                    self.set_layer_quant(model=module, module_name_list=module_name_list, group_list=group_list, group_ignore=group_ignore, quant_level='per_block', weight_quant=weight_quant, act_quant=act_quant, prefix=full_name+".")


    def set_layer_bit(self, model=None, n_bit=None, module_name_list=[], group_list=[], quant_level='per_layer', bit_type='weight', prefix=""):
        '''
        Progressively set bit of the the groups or layers, which is different from the func in the quant_error.py.
        Selectivly quantize some layers of groups to low bit
        bit_type: 'weight' of 'act', we can only quantize weight or act at once
        '''
        if quant_level=='per_layer':
            for name, module in model.named_children():
                full_name = prefix + name if prefix else name
                if bit_type == 'weight':
                    if isinstance(module, WeightQuantizer):
                        for module_name in module_name_list:
                            if  module_name in module.module_name:
                                module.bitwidth_refactor(n_bit)
                                torch.cuda.empty_cache()
                                # logger.info(f"{full_name}: weight_bit={n_bit}")
                    else:
                        self.set_layer_weight_bit(model=module, n_bit=n_bit, module_name_list=module_name_list, group_list=group_list, quant_level='per_layer', bit_type=bit_type, prefix=full_name+".")
                elif bit_type == 'act':
                    if isinstance(module, ActQuantizer):
                        for module_name in module_name_list:
                            if  module_name in module.module_name:
                                module.bitwidth_refactor(n_bit)
                                torch.cuda.empty_cache()
                                # logger.info(f"{full_name}: act_bit={n_bit}")
                    else:
                        self.set_layer_bit(model=module, n_bit=n_bit, module_name_list=module_name_list, group_list=group_list, quant_level='per_layer', bit_type=bit_type, prefix=full_name+".")

        elif quant_level=='per_group':
            for name, module in model.named_children():
                full_name = prefix + name if prefix else name
                if bit_type == 'weight':
                    if isinstance(module, WeightQuantizer):
                        for group_name in group_list:
                            if group_name in module.module_name:
                                module.bitwidth_refactor(n_bit)
                                torch.cuda.empty_cache()
                                logger.info(f"{full_name}: weight_quant={n_bit}")
                    else:
                        self.set_layer_bit(model=module, n_bit=n_bit, module_name_list=module_name_list, group_list=group_list, quant_level='per_group', bit_type=bit_type, prefix=full_name+".")
                if bit_type == 'act':
                    if isinstance(module, ActQuantizer):
                        for group_name in group_list:
                            if group_name in module.module_name:
                                module.bitwidth_refactor(n_bit)
                                torch.cuda.empty_cache()
                                logger.info(f"{full_name}: act_quant={n_bit}")
                    else:
                        self.set_layer_bit(model=module, n_bit=n_bit, module_name_list=module_name_list, group_list=group_list, quant_level='per_group', bit_type=bit_type, prefix=full_name+".")
        
        elif quant_level=='reset':
            for name, module in model.named_children():
                full_name = prefix + name if prefix else name
                if bit_type == 'weight':
                    if isinstance(module, WeightQuantizer):
                                module.bitwidth_refactor(n_bit)
                                torch.cuda.empty_cache()
                                # logger.info(f"{full_name}: weight_quant={n_bit}")
                    else:
                        self.set_layer_bit(model=module, n_bit=n_bit, module_name_list=module_name_list, group_list=group_list, quant_level='reset', bit_type=bit_type, prefix=full_name+".")
                if bit_type == 'act':
                    if isinstance(module, ActQuantizer):
                                module.bitwidth_refactor(n_bit)
                                torch.cuda.empty_cache()
                                # logger.info(f"{full_name}: act_quant={n_bit}")
                    else:
                        self.set_layer_bit(model=module, n_bit=n_bit, module_name_list=module_name_list, group_list=group_list, quant_level='reset', bit_type=bit_type, prefix=full_name+".")


    def load_bitwidth_config(self, model, bit_config, bit_type, prefix=''):
        '''
        please pass the bit_config of weight and act seperatly
        '''
        for name, module in model.named_children():
            full_name = prefix + name if prefix else name

            if isinstance(module, QuantLayer):
                if full_name in bit_config.keys():
                    if bit_type == 'weight':
                        module.weight_quantizer.bitwidth_refactor(bit_config[full_name])
                        if hasattr(module, 'weight_quantizer_0'):
                            module.weight_quantizer_0.bitwidth_refactor(bit_config[full_name])
                        logger.info(f"{full_name}: the w_bit = {bit_config[full_name]}")

                    elif bit_type == 'act':
                        module.act_quantizer.bitwidth_refactor(bit_config[full_name])
                        if hasattr(module, 'act_quantizer_0'):
                            module.act_quantizer_0.bitwidth_refactor(bit_config[full_name])      
                        logger.info(f"{full_name}: the a_bit = {bit_config[full_name]}")

                    torch.cuda.empty_cache()

            else:
                self.load_bitwidth_config(model=module, bit_config=bit_config, bit_type=bit_type, prefix=full_name+".")


    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
