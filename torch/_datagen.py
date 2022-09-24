import torch

def data_gen(spec):
    return torch.rand(*spec.shape, dtype=spec.dtype, device='cuda')
