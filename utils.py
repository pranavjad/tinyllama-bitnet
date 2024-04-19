from torch import nn
from transformers.models.llama.modeling_llama import *

def activation_quant(x):
    """ Per−token quantization to 8 bits. No grouping is needed for quantization.
    Args:
    x: an activation tensor with shape [n, d]
    Returns:
    y: a quantized activation tensor with shape [n, d]
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y
def weight_quant(w, return_scale=False):
    """ Per−tensor quantization to 1.58 bits. No grouping is needed for quantization.
    Args:
    w: a weight tensor with shape [d, k]
    Returns:
    u: a quantized weight with shape [d, k]
    """
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1)
    if return_scale:
        return u, scale
    return (u / scale)

class BitLinear(nn.Linear):
    def __init__(self, *args, weight_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.RMSNorm = LlamaRMSNorm(self.in_features)
        self.weight_scale = weight_scale
    """
    This is only for training, and kernel optimization is needed for efficiency.
    """
    def forward(self, x):
        """
        Args:
        x: an input tensor with shape [n, d]
        Returns:
        y: an output tensor with shape [n, d]
        """
        if self.training:
            w = self.weight # a weight tensor with shape [d, k]
            x_norm = self.RMSNorm(x)
            # A trick for implementing Straight−Through−Estimator (STE) using detach()
            x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
            w_quant = w + (weight_quant(w) - w).detach()
            y = F.linear(x_quant, w_quant)
        else: # Assume weights are already quantized
            x_norm = self.RMSNorm(x)
            x_quant = activation_quant(x_norm)
            y = F.linear(x_quant, self.weight) / self.weight_scale
        return y


def convert_to_bitnet(model, copy_weights=False, inference=False):
    """ Converts a LLamaForCausalLM model to bitnet architecture. 
    There are two steps to achieve this according to the released training details:
    1. Replace all nn.Linear in attention and SwiGLU with BitLinear
    2. Remove RMSNorm before attention and SwiGLU because BitLinear has built-in RMSNorm

    Args:
    model: A LLamaForCausalLM model
    copy_weights: Boolean value indicating whether to copy the weights of the linear layers to Bitnet layers. Useful for continued
    pretraining.
    inference: Boolean value indicating if the model is being converted for inference. If true, the weights will be pre-quantized.
    """
    for name, module in model.named_modules():
        # Replace linear layers with BitNet
        if isinstance(module, LlamaSdpaAttention) or isinstance(module, LlamaMLP):
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Linear):
                    if inference: # pre-quantize weights
                        q_weight, weight_scale = weight_quant(child_module.weight, return_scale=True)
                        bitlinear = BitLinear(child_module.in_features, child_module.out_features,
                                              child_module.bias is not None, weight_scale=weight_scale).to(device="cuda:0")
                        bitlinear.weight = nn.Parameter(q_weight)
                    else:
                        bitlinear = BitLinear(child_module.in_features, child_module.out_features, child_module.bias is not None).to(device="cuda:0")                        
                        if copy_weights:
                            bitlinear.weight = child_module.weight
                            if child_module.bias is not None:
                                bitlinear.bias = child_module.bias
                    setattr(module, child_name, bitlinear)
        # Remove layernorms
        elif isinstance(module, LlamaDecoderLayer):
            for child_name, child_module in module.named_children():
                if isinstance(child_module, LlamaRMSNorm):
                    setattr(module, child_name, nn.Identity().to(device="cuda:0"))