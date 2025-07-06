import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from quant_utils import QuantLinear
from transformers.modeling_utils import Conv1D 
from peft import get_peft_model, LoraConfig, PeftModel
import warnings
from typing import Dict
from defaults import SUPPORTED_BWS
# GPT-2 Config:
# GPT2Model(
#   (wte): Embedding(50257, 768)
#   (wpe): Embedding(1024, 768)
#   (drop): Dropout(p=0.1, inplace=False)
#   (h): ModuleList(
#     (0-11): 12 x GPT2Block(
#       (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#       (attn): GPT2Attention(
#         (c_attn): Conv1D(nf=2304, nx=768)
#         (c_proj): Conv1D(nf=768, nx=768)
#         (attn_dropout): Dropout(p=0.1, inplace=False)
#         (resid_dropout): Dropout(p=0.1, inplace=False)
#       )
#       (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
#       (mlp): GPT2MLP(
#         (c_fc): Conv1D(nf=3072, nx=768)
#         (c_proj): Conv1D(nf=768, nx=3072)
#         (act): NewGELUActivation()
#         (dropout): Dropout(p=0.1, inplace=False)
#       )
#     )
#   )
#   (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
# )

def _recursive_patch_quantization(module, prefix=""):
    """
    Walks through a module and replaces Conv1D layers with QuantLinear layers.

    Args:
        module (nn.Module): The module to patch.
        prefix (str): The prefix for layer names.
    """
    for name, child in module.named_children():
        # Construct the full name of the child module
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, Conv1D):
            in_features = child.weight.shape[0]
            out_features = child.weight.shape[1]
            device = child.weight.device
            q_layer = QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=(child.bias is not None),
                device=device
            )
            # Copy the weights and bias from the original layer
            q_layer.weight.data.copy_(child.weight.data.t())
            if child.bias is not None:
                q_layer.bias.data.copy_(child.bias.data)
            
            setattr(module, name, q_layer)
            
            # Replace the original layer with the new quantized layer
            setattr(module, name, q_layer)
        else:
            # If it's not a Linear layer, recurse deeper
            _recursive_patch_quantization(child, prefix=full_name)

def patch_gpt2_with_quantization(model: AutoModelForCausalLM):
    """
    Applies quantization to a GPT-2 model by replacing all nn.Linear layers with QuantLinear layers.

    Args:
        model (AutoModelForCausalLM): The pre-trained GPT-2 model to be patched.
    Returns:
        AutoModelForCausalLM: The patched model with QuantLinear layers.
    """
    _recursive_patch_quantization(model)
    return model

class SwitchableQuantLoRAModel:
    """A wrapper to manage the PEFT model and switchable precision / adapters """
    def __init__(self, model: PeftModel, target_modules: list[str]):
        self.model = model
        self.target_modules = target_modules
        self.quant_named_modules = {}
        for name, module in model.named_modules():
            module_name = name.removeprefix('base_model.model.').removesuffix('.base_layer')
            if module_name in target_modules and isinstance(module, QuantLinear):
                self.quant_named_modules[module_name] = module
        assert len(self.quant_named_modules) == len(self.target_modules), 'some target modules are not quantized'

    def set_config(self, bitmap: Dict[str, int]):
        """
        Sets the active adapters based on quantization config (bitmap), also change bitwidth in QuantLinear layers
        """
        if len(bitmap) != len(self.target_modules):
            raise ValueError("Bitmap length must match the number of target layers.")

        # Determine the list of active LoRA adapters
        active_adapters = []
        for module_full_name, bw in bitmap.items():
            adapter_name = module_full_name.replace('.', '-') + f'-{bw}'
            active_adapters.append(adapter_name)
        self.model.base_model.set_adapter(active_adapters)
        for module_full_name, bw in bitmap.items():
            self.quant_named_modules[module_full_name].set_bit_width(bw)

    def check_config(self, bitmap: Dict[str, int]):
        """
        check model's config are the same as bitmap, also checks require_grad
        """
        model_adapters = self.model.get_model_status().active_adapters
        active_adapters = []
        for module_full_name, bw in bitmap.items():
            adapter_name = module_full_name.replace('.', '-') + f'-{bw}'
            active_adapters.append(adapter_name)
        if not set(model_adapters) == set(active_adapters):
            raise AssertionError('model\'s active adapters are not the same as bitmap specified')
        # Check that QuantLinear layers have the right bit-width
        for module_full_name, bw in bitmap.items():
            # Verify the bit_width of the underlying layer
            actual_bw = self.quant_named_modules[module_full_name].bit_width
            assert actual_bw == bw, f"Actual bit width={actual_bw} in {module_full_name} not the same in bitmap:{bw}"
        # Check active adapters require_grad = True, otherwise False
        for name, require_grad in self.model.get_model_status().requires_grad.items():
            if (require_grad and (name not in active_adapters) ) or (not require_grad and (name in active_adapters)):
                warnings.warn(f'Inconsistent require_grad: {name}: {require_grad}')
        return
        
    def __call__(self, *args, **kwargs):
        """Pass calls through to the underlying model."""
        return self.model(*args, **kwargs)

def patch_gpt2_with_adaptive_adapters(model: AutoModelForCausalLM, supported_bws = SUPPORTED_BWS) -> SwitchableQuantLoRAModel:
    """
    Patch GPT2 with multiple adapters in each layer, will patch $supported_bws adapters (e.g. 2/4/8 bits) for each linear layers

    Args:
        model (AutoModelForCausalLM): The GPT2Model already patched with QuantLinear layers
        supported_bws (List[str]): list of supported bitwidth
    """
    print('Patching model with adapters, will enable adaptive adapters based on quant level')
    # Freeze all parameters of the base quantized model
    for param in model.parameters():
        param.requires_grad = False
    print(f"- Froze base model, will add adapters for supported bit-widths: {supported_bws}")

    # target_modules: all QuantLinear layers in the model.
    target_modules = [
        name for name, module in model.named_modules()
        if isinstance(module, QuantLinear)
    ]
    print(f"- {len(target_modules)} target layers for LoRA")

    base_config = LoraConfig(r=8, target_modules=["c_attn"]) # A dummy adapter, will be deleted
    peft_model = get_peft_model(model, base_config, "base_adapter")

    # loop and add a specific adapter for each layer and bit-width
    for module_full_name in target_modules:
        for bw in supported_bws:
            adapter_name = module_full_name.replace('.', '-') + f'-{bw}'
            config = LoraConfig(
                r=16, # In a real scenario, rank could be a function of bit-width
                lora_alpha=16,
                target_modules=[module_full_name]
            )
            peft_model.add_adapter(adapter_name, config)
            peft_model.set_adapter(adapter_name)
    peft_model.disable_adapter()
    peft_model.delete_adapter('base_adapter')
    print(f"- Added {len(peft_model.peft_config)} total adapters.")
    assert len(peft_model.peft_config) == (len(supported_bws)*len(target_modules))

    return SwitchableQuantLoRAModel(peft_model, target_modules)
    


    