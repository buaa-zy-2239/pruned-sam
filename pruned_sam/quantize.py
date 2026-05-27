import torch
import torch.nn as nn


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features))
    
    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)


class SimpleW8A8Quantizer:
    def __init__(self):
        pass
    
    def quantize_weight(self, weight):
        return weight
    
    def dequantize_weight(self, q_weight, scale, zero_point):
        return q_weight


def build_quantized_pruned_sam(config_name, checkpoint=None):
    from .build import build_pruned_sam
    return build_pruned_sam(config_name, checkpoint)


def quantize_model(model, config=None):
    return model


def get_quantization_info(model):
    return {"num_layers": 0, "quantized_params": 0}


def benchmark_quantized_model(model, data_loader, device='cpu'):
    return {"avg_time": 0.0, "throughput": 0.0}


def save_quantized_model(model, path):
    torch.save(model.state_dict(), path)


def load_quantized_model(config_name, path):
    from .build import build_pruned_sam
    model = build_pruned_sam(config_name)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    return model