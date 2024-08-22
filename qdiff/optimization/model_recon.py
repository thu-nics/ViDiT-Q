import torch
import logging

from qdiff.models.quant_layer import QuantLayer
from qdiff.models.quant_block import BaseQuantBlock
from qdiff.models.quant_model import QuantModel
from qdiff.optimization.block_recon import block_reconstruction
from qdiff.optimization.layer_recon import layer_reconstruction


logger = logging.getLogger(__name__)

def model_reconstruction(model, module, calib_data, config, param_types, opt_target, prefix=""):
    # INFO: due to that the layer_reconstruct and block_reconstruct need to feed in the **quantized_whole_model**
    # while the model is used for recursively conduct reconstruction
    # names = []
    # modules = []
    # for name, module in model.named_children():
        # names.append(name)
        # modules.append(module)

    # INFO: model is always the whole module, iter through the 'module'
    for name, module_ in module.named_children():
        full_name = prefix + name if prefix else name
        # logger.info(f"{name} {isinstance(module, BaseQuantBlock)}")
        torch.cuda.empty_cache()
        # DIRTY: hard coded output_blocks as last module
        # if name == 'output_blocks':
            # logger.info("Finished calibrating input and mid blocks, saving temporary checkpoint...")
            # torch.save(quantized_whole_model.state_dict(), os.path.join(outpath, "ckpt.pth"))
            # return None

        # if name.isdigit() and int(name) >= 9:
        #     logger.info(f"Saving temporary checkpoint at {name}...")
        #     torch.save(quantized_whole_model.state_dict(), os.path.join(outpath, "ckpt.pth"))

        # layer reconstruction
        if isinstance(module_, QuantLayer):
            if module_.ignore_reconstruction is True:
                logger.info('Ignore {} reconstruction of layer {}'.format(opt_target, full_name))
                continue
            elif module_.weight_quantizer.module_name.split('.')[0] in model.fp_layer_list:
                logger.info('Ignore {} reconstruction of layer {}'.format(opt_target, full_name))
                continue
            # due to differnet input size, skip opt for kv_linear
            elif 'kv_linear' in module_.weight_quantizer.module_name:
                logger.info('Ignore {} reconstruction of layer {}'.format(opt_target, full_name))
                continue
            else:
                logger.info('{} Reconstruction for layer {}'.format(opt_target, full_name))
                layer_reconstruction(model, module_, calib_data, config, param_types, opt_target)

        # module reconstruction
        # elif isinstance(module_, BaseQuantBlock):
        #     if module_.ignore_reconstruction is True:
        #         logger.info('Ignore {} reconstruction of block {}'.format(opt_target, full_name))
        #         continue
        #     else:
        #         logger.info('{} Reconstruction for block {}'.format(opt_target, full_name))
        #         block_reconstruction(model, module_, calib_data, config, param_types, opt_target)

        else:
            model_reconstruction(model, module_, calib_data, config, param_types, opt_target, prefix=full_name+'.')



