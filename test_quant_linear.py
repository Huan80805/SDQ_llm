import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from transformers.modeling_utils import Conv1D
from src import QuantLinear, InferenceQuantLinear, patch_gpt2_with_quantization

def test_quant_linear_equivalence():
    print("Testing QuanatLinear functionality")
    torch.manual_seed(0)
    q_layer = QuantLinear(30, 30, bit_width=0)
    x = torch.randn(2, 30) # random input tensor
    y_fp = q_layer(x)
    print("- FP output:\n", y_fp)

    for bit_width in [8, 6, 4, 2]:
        q_layer.set_bit_width(bit_width)
        y_q = q_layer(x)
        assert (y_fp - y_q).abs().max() < 1, f"Max absolute difference exceeds tolerance with y:{y_fp}, y_q:{y_q}"
        assert y_fp.size() == y_q.size()
        print(f"[BW: {bit_width}]")
        print("- Quantized output:\n", y_q)

def test_inference_quant_linear():
    print("Testing InferenceQuanatLinear functionality")
    torch.manual_seed(0)
    q_layer = QuantLinear(32, 32, bit_width=0, bias=True)
    x = torch.randn(2, 32) # random input tensor
    y_fp = q_layer(x)
    print("- FP output:\n", y_fp)
    for bit_width in [8, 4, 2]:
        # FP32 Reference
        inference_layer = InferenceQuantLinear(32, 32, bias=True, bit_width=bit_width)
        inference_layer.quantize_and_pack(q_layer.weight.data, q_layer.bias.data)
        
        q_layer.set_bit_width(bit_width)
        y_q = q_layer(x)
        y_iq = inference_layer(x)
        
        assert (y_fp - y_iq).abs().max() < 1, f"Max absolute difference exceeds tolerance with y:{y_fp}, y_q:{y_q}"
        assert (y_q - y_iq).abs().max() < 0.1, f"Max absolute difference exceeds tolerance with y_q:{y_q}, y_iq:{y_q}"
        assert y_fp.shape == y_q.shape == y_iq.shape, "Output shapes do not match."
        print(f"[BW: {bit_width}]")
        print("- Quantized output:\n", y_q)
        print("- Truly Quantized output:\n", y_iq)

def test_patching():
    """
    Tests that the patching function correctly replaces all Conv1D (Linear) layers in a GPT-2 model with QuantLinear layers.
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