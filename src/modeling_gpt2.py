import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from .quant_utils import QuantLinear, InferenceQuantLinear
from .defaults import SUPPORTED_BWS
from transformers.modeling_utils import Conv1D 
from peft import get_peft_model, LoraConfig, PeftModel
import warnings
from typing import Dict

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

def patch_gpt2_with_quantization(model: AutoModelForCausalLM):
    """
    Applies quantization to a GPT-2 model by replacing all nn.Linear layers with QuantLinear layers.

    Args:
        model (AutoModelForCausalLM): The pre-trained GPT-2 model to be patched.
    Returns:
        AutoModelForCausalLM: The patched model with QuantLinear layers.
    """
    # Get a list of all linear layer names in the model
    source_modules = {name: module for name, module in model.named_modules() if isinstance(module, Conv1D)}
    for name, child_mod in source_modules.items():
        # Navigate to the parent module
        parent_name, child_name = name.rsplit('.', 1)
        in_features = child_mod.weight.shape[0]
        out_features = child_mod.weight.shape[1]
        parent_mod = model.get_submodule(parent_name)
        
        # Create the QuantLinear layer
        device = child_mod.weight.device
        q_layer = QuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=(child_mod.bias is not None),
            device=device
        )

        # Copy the weights and bias from the original layer
        q_layer.weight.data.copy_(child_mod.weight.data.t())
        if child_mod.bias is not None:
            q_layer.bias.data.copy_(child_mod.bias.data)
        # Replace the layer in the new model
        setattr(parent_mod, child_name, q_layer)
    return model

class SwitchableQuantLoRAModel:
    """A wrapper to manage the PEFT model and switchable precision / adapters """
    def __init__(self, model: PeftModel, target_modules: list[str]):
        self.model = model
        self.target_modules = target_modules
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
            if bw != 0:
                active_adapters.append(adapter_name)
            # Get QuantLinear layer, now module_full_name is a adapter layer, have to get base_layer
            qunat_linear_layer = self.model.base_model.get_submodule(module_full_name + ".base_layer")
            qunat_linear_layer.set_bit_width(bw)

        self.model.base_model.set_adapter(active_adapters)

    def check_config(self, bitmap: Dict[str, int]):
        """
        check model's config are the same as bitmap, also checks require_grad
        """
        model_adapters = self.model.active_adapters
        active_adapters = []
        for module_full_name, bw in bitmap.items():
            if bw != 0:
                adapter_name = module_full_name.replace('.', '-') + f'-{bw}'
                active_adapters.append(adapter_name)
        assert set(model_adapters) == set(active_adapters), 'model\'s active adapters are not the same as bitmap specified'
        # Check that QuantLinear layers have the right bit-width
        for module_full_name, bw in bitmap.items():
            # Verify the bit_width of the underlying layer
            qunat_layer_base = self.model.base_model.get_submodule(module_full_name + ".base_layer")
            actual_bw = qunat_layer_base.bit_width
            assert actual_bw == bw, f"Actual bit width={actual_bw} in {module_full_name} not the same in bitmap:{bw}"
        # Check active adapters require_grad = True, otherwise False
        for name, require_grad in self.model.get_model_status().requires_grad.items():
            if (require_grad and (name not in active_adapters) ) or (not require_grad and (name in active_adapters)):
                warnings.warn(f'Inconsistent require_grad: {name}: {require_grad}')
        return
        
    def __call__(self, *args, **kwargs):
        """Pass calls through to the underlying model."""
        return self.model(*args, **kwargs)
    
def create_quant_model_for_inference(model: PeftModel, bitmap_config:Dict[str, int]):
    """
    Converts into a truly quantized inference model by replacing its linear layers in-place.
    """
    # Get a list of all linear layer names in the model
    source_modules = {name: module for name, module in model.base_model.model.named_modules() if isinstance(module, QuantLinear)}
    assert len(source_modules) > 0, "Empty source_modules"
    for name, child_mod in source_modules.items():
        # Navigate to the parent module
        parent_name, child_name = name.rsplit('.', 1)
        bit_width = bitmap_config[name.removesuffix('.base_layer')]
        if bit_width == 0: continue
        parent_mod = model.get_submodule(parent_name)
        device = child_mod.weight.device
        # Create the InferenceQuantLinear layer
        inf_layer = InferenceQuantLinear(
            child_mod.in_features,
            child_mod.out_features,
            bias=(child_mod.bias is not None),
            bit_width=bit_width
        ).to(device)
        
        # Use the weights from the source module to quantize and pack the weights for the new inference layer.
        inf_layer.quantize_and_pack(child_mod.weight.data, child_mod.bias.data if child_mod.bias is not None else None)
        # Replace the original QuantLinear layer with the new, memory-efficient InferenceQuantLinear layer
        setattr(parent_mod, child_name, inf_layer)

    return model

def patch_gpt2_with_adaptive_adapters(model: AutoModelForCausalLM, supported_bws = SUPPORTED_BWS, verbose = False) -> SwitchableQuantLoRAModel:
    """
    Patch GPT2 with multiple adapters in each layer, will patch $supported_bws adapters (e.g. 2/4/8 bits) for each linear layers

    Args:
        model (AutoModelForCausalLM): The GPT2Model already patched with QuantLinear layers
        supported_bws (List[str]): list of supported bitwidth
    """
    if verbose:
        print('- Patching model with adapters, will enable adaptive adapters based on quant level')
    # Freeze all parameters of the base quantized model
    for param in model.parameters():
        param.requires_grad = False
    if verbose:
        print(f"- Froze base model, will add adapters for supported bit-widths: {supported_bws}")

    # target_modules: all QuantLinear layers in the model.
    target_modules = [
        name for name, module in model.named_modules()
        if isinstance(module, QuantLinear)
    ]
    if verbose:
        print(f"- {len(target_modules)} target layers for LoRA")
    base_config = LoraConfig(r=8, target_modules=["c_attn"]) # A dummy adapter, will be deleted
    peft_model = get_peft_model(model, base_config, "base_adapter")
    get_rank_by_bw = {8: 16, 4:32, 2: 64}
    # loop and add a specific adapter for each layer and bit-width
    for module_full_name in target_modules:
        for bw in supported_bws:
            adapter_name = module_full_name.replace('.', '-') + f'-{bw}'
            config = LoraConfig(
                r=get_rank_by_bw[bw],
                lora_alpha=get_rank_by_bw[bw]*2,
                target_modules=[module_full_name]
            )
            peft_model.add_adapter(adapter_name, config)
            peft_model.set_adapter(adapter_name)
    peft_model.disable_adapter()
    peft_model.delete_adapter('base_adapter')
    if verbose:
        print(f"- Added {len(peft_model.peft_config)} total adapters.")
    assert len(peft_model.peft_config) == (len(supported_bws)*len(target_modules))

    return SwitchableQuantLoRAModel(peft_model, target_modules)
    


    