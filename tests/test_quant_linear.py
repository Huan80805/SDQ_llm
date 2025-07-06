import torch
import torch.nn as nn
import pytest
from transformers import GPT2Config, GPT2Model
from transformers.modeling_utils import Conv1D
import sys
sys.path.append('src')

from quant_utils import QuantLinear
from modeling_gpt2 import patch_gpt2_with_quantization

def test_quant_linear_equivalence():
    for bit_width in [8, 6, 4, 2]:
        torch.manual_seed(0)
        # Create a QuantLinear layer with a bit-width of 4
        q_layer = QuantLinear(30, 30, bit_width=bit_width)
        # Create a random input tensor
        x = torch.randn(2, 30)

        # Calculate the full-precision output for comparison
        y_fp = nn.functional.linear(x, q_layer.weight, q_layer.bias)
        # Calculate the output from our quantized layer
        y_q = q_layer(x)

        # The difference should be small (quantization introduces error)
        assert (y_fp - y_q).abs().max() < 1, f"Max absolute difference exceeds tolerance with y:{y_fp}, y_q:{y_q}"
        assert y_fp.size() == y_q.size()
        print(f"[BW: {bit_width}]")
        print(y_fp, y_q)

def test_patching():
    """
    Tests that the patching function correctly replaces all nn.Linear
    layers in a GPT-2 model with QuantLinear layers.
    """
    torch.manual_seed(0)
    config = GPT2Config(n_layer=2, n_embd=64, n_head=4)
    model = GPT2Model(config)
    
    patched_model = patch_gpt2_with_quantization(model)
    found_quant_linear = False
    for name, module in patched_model.named_modules():
        # Assert that no standard nn.Linear layers are left
        assert not isinstance(module, Conv1D), f"Found an unpatched Conv1D layer: {name}"  
        if isinstance(module, QuantLinear):
            found_quant_linear = True

    assert found_quant_linear, "Patching failed: no QuantLinear layers were found."