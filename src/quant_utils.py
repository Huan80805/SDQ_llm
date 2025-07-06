import torch
import torch.nn as nn
import torch.nn.functional as F
from defaults import SUPPORTED_BWS
class QuantLinear(nn.Linear):
    """
    A drop-in replacement for torch.nn.Linear that supports dynamic quantization of its weights.
    This layer maintains a full-precision (FP32) master copy of the weights and quantizes them on-the-fly during the forward pass. 

    Args:
        bit_width (int): The default quantization bw
    """
    def __init__(self, in_features, out_features, bit_width=8, bias=True,
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)
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
        w_q, scale, zero_pt = self._quantize_tensor(self.weight, bits=bw)
        w_dequantized = self._dequantize_tensor(w_q, scale, zero_pt)
        
        return F.linear(x, w_dequantized, self.bias)

    @staticmethod
    @torch.no_grad()
    def _quantize_tensor(t: torch.Tensor, bits: int, symmetric: bool = True):
        """
        Performs symmetric/asymmetric uniform quantization on a weight tensor.
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
    
    @staticmethod
    @torch.no_grad()
    def _dequantize_tensor(q_t: torch.Tensor, scale: torch.Tensor, zero_pt: torch.Tensor) -> torch.Tensor:
        """De‑quantize a tensor back to FP32 given its scale & zero‑point."""
        return (q_t.float() - zero_pt) * scale