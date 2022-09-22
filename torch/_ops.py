import inspect
import torch

def builtin(f):
    assert inspect.isbuiltin(f)
    return f

linear = builtin(torch.nn.functional.linear)
dropout = builtin(torch._VF.dropout)
dropout_ = builtin(torch._VF.dropout_)
matmul = builtin(torch.matmul)
layernorm = builtin(torch.nn.functional.layer_norm)

def reshape(x, shape):
    return x.view(shape)

def transpose(x, dims):
    return x.permute(*dims)

def softmax(x, dim):
    return x.softmax(dim)

