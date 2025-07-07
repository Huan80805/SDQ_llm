import torch
import torch.nn as nn
import torch.nn.functional as F
from .defaults import SUPPORTED_BWS
class QuantLinear(nn.Linear):
    """
    Replacement for torch.nn.Linear that supports dynamic quantization of its weights.
    Maintaining a full-precision (FP32) master copy of the weights and quantizes them on-the-fly during the forward pass. 

    Args:
        bit_width (int): The default quantization bw
    """
    def __init__(self, in_features, out_features, bit_width=8, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.bit_width = bit_width

    def set_bit_width(self, new_bits: int):
        """Allows dynamically changing the quantization bit-width."""
        self.bit_width = new_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bw = self.bit_width

        # No quant
        if bw == 0:
            return F.linear(x, self.weight, self.bias)
        elif bw not in SUPPORTED_BWS: raise ValueError(f"bit width {bw} not supported")
        # Do quant
        w_q, scale, zero_pt = quantize_tensor(self.weight, bits=bw)
        w_dequantized = dequantize_tensor(w_q, scale, zero_pt)
        
        return F.linear(x, w_dequantized, self.bias)

@torch.no_grad()
def quantize_tensor(t: torch.Tensor, bits: int, symmetric: bool = True):
    """
    Performs symmetric/asymmetric uniform (simulated with int8) quantization on a weight tensor. 
    scale is determined by rowwise max/min
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the
        quantized weight tensor, corresponding FP32 scaling factors (out_features, 1), and zero point.
    """
    qmax = 2 ** (bits - 1) - 1 if symmetric else 2 ** bits - 1

    if symmetric:
        max_abs = t.abs().amax(dim=1, keepdim=True)  # [out, 1]
        scale = (max_abs + 1e-8) / qmax
        zero_pt = 0
        q_t = torch.round(t / scale).clamp(-qmax, qmax).to(torch.int8)
    else:
        t_min = t.amin(dim=1, keepdim=True)
        t_max = t.amax(dim=1, keepdim=True)
        scale = (t_max - t_min) / qmax
        zero_pt = (-t_min / scale).round()
        q_t = ((t / scale) + zero_pt).round().clamp(0, qmax).to(torch.uint8)

    return (q_t, scale, zero_pt)
    
@torch.no_grad()
def dequantize_tensor(q_t: torch.Tensor, scale: torch.Tensor, zero_pt: torch.Tensor) -> torch.Tensor:
    """De‑quantize a tensor back to FP32 given its scale & zero‑point."""
    return (q_t.float() - zero_pt) * scale


class InferenceQuantLinear(nn.Module):
    """
    A truly quantized linear layer for inference that stores packed weights to minimize VRAM usage. It does not store the original FP32 weights.
    """
    def __init__(self, in_features, out_features, bias=True, bit_width=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width

        # Determine the packed dimension size
        if bit_width == 8:
            packed_in_features = in_features
        elif bit_width == 4:
            packed_in_features = in_features // 2
        elif bit_width == 2:
            packed_in_features = in_features // 4
        else:
            raise ValueError("Supported bit-widths are 2, 4, and 8.")
            
        # Register buffers for packed weights and scales, not parameters
        self.register_buffer('packed_weight', torch.empty(out_features, packed_in_features, dtype=torch.int8))
        self.register_buffer('scale', torch.empty(out_features, 1, dtype=torch.float32))
        
        if bias:
            self.register_buffer('bias', torch.empty(out_features, dtype=torch.float32))
        else:
            self.bias = None

    def quantize_and_pack(self, weight_fp32, bias_fp32):
        """ Quantizes, packs, and stores the weights and bias. The FP32 tensors can be discarded after this. """
        if self.bias is not None and bias_fp32 is not None:
            self.bias.copy_(bias_fp32)
        
        q_t, scale, _ = quantize_tensor(weight_fp32, bits=self.bit_width, symmetric=True)
        self.scale.copy_(scale)

        if self.bit_width == 8:
            self.packed_weight.copy_(q_t)
        elif self.bit_width == 4:
            self.packed_weight.copy_(_pack_4bit(q_t))
        elif self.bit_width == 2:
            self.packed_weight.copy_(_pack_2bit(q_t))

    def forward(self, x):
        # Unpack on-the-fly
        if self.bit_width == 8:
            w_q = self.packed_weight
        elif self.bit_width == 4:
            w_q = _unpack_4bit(self.packed_weight, self.in_features)
        elif self.bit_width == 2:
            w_q = _unpack_2bit(self.packed_weight, self.in_features)
        else:
            raise NotImplementedError("Unsupported bit-width for unpacking.")

        w_dequantized = w_q.to(x.dtype) * self.scale
        return F.linear(x, w_dequantized, self.bias)
    
# --- Packing and Unpacking Utilities ---

@torch.no_grad()
def _pack_4bit(w_q_symmetric):
    """ Packs a tensor of symmetric 4-bit values [-8, 7] into a packed int8 tensor. """
    # Shift from [-8, 7] to [0, 15] to be unsigned
    w_q_unsigned = w_q_symmetric.to(torch.uint8) + 8
    
    if w_q_unsigned.shape[-1] % 2 != 0:
        raise ValueError("The last dimension for 4-bit packing must be even.")

    # Pack two 4-bit values into one int8
    w_packed = (w_q_unsigned[..., ::2] & 0x0F) | (w_q_unsigned[..., 1::2] << 4)
    return w_packed

@torch.no_grad()
def _unpack_4bit(w_packed, original_last_dim):
    """ Unpacks an int8 tensor into symmetric 4-bit values. """
    w_unpacked = torch.empty(
        (*w_packed.shape[:-1], original_last_dim),
        dtype=torch.int8,
        device=w_packed.device
    )
    
    # Unpack from low and high nibbles
    w_unpacked[..., ::2] = w_packed & 0x0F
    w_unpacked[..., 1::2] = (w_packed >> 4) & 0x0F
    
    # Shift back from [0, 15] to [-8, 7]
    return w_unpacked - 8

@torch.no_grad()
def _pack_2bit(w_q_symmetric):
    """ Packs a tensor of symmetric 2-bit values [-2, 1] into a packed int8 tensor. """
    # Shift from [-2, 1] to [0, 3] to be unsigned
    w_q_unsigned = w_q_symmetric.to(torch.uint8) + 2

    if w_q_unsigned.shape[-1] % 4 != 0:
        raise ValueError("The last dimension for 2-bit packing must be divisible by 4.")
    
    # Pack four 2-bit values into one int8
    w_packed = (w_q_unsigned[..., ::4] & 0x03) | \
               ((w_q_unsigned[..., 1::4] & 0x03) << 2) | \
               ((w_q_unsigned[..., 2::4] & 0x03) << 4) | \
               ((w_q_unsigned[..., 3::4] & 0x03) << 6)
    
    return w_packed

@torch.no_grad()
def _unpack_2bit(w_packed, original_last_dim):
    """ Unpacks an int8 tensor into symmetric 2-bit values. """
    w_unpacked = torch.empty(
        (*w_packed.shape[:-1], original_last_dim),
        dtype=torch.int8,
        device=w_packed.device
    )

    # Unpack from the four 2-bit segments
    w_unpacked[..., ::4] = w_packed & 0x03
    w_unpacked[..., 1::4] = (w_packed >> 2) & 0x03
    w_unpacked[..., 2::4] = (w_packed >> 4) & 0x03
    w_unpacked[..., 3::4] = (w_packed >> 6) & 0x03
    
    # Shift back from [0, 3] to [-2, 1]
    return w_unpacked - 2