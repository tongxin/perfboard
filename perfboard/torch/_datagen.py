import torch

def torch_dtype(dtype: str):
    str2type = {
        'float': torch.float,
        'float32': torch.float,
        'float64': torch.float64,
        'float16': torch.float16,
        'half': torch.half,
        'bfloat': torch.bfloat16,
        'int': torch.int,
        'int64': torch.long,
        'int32': torch.int,
        'int16': torch.short,
        'short': torch.short,
        'int8': torch.int8,
        'bool': torch.bool
    }
    dtype = str2type.get(dtype)
    assert dtype is not None, f'Invalid input dtype: {dtype}'
    return dtype

def data_gen(spec):
    dtype = torch_dtype(spec.dtype)
    return torch.rand(*spec.shape, dtype=dtype, device='cuda')
