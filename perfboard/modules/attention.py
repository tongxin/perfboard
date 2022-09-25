import math

from ..common.tensor import DataSpec
from ..benchmark import Module, TestModule

class AttentionOutput(Module):
    TESTS = [
        (768, 0.1, 1e-12),
        (1024, 0.1, 1e-12)
    ]

    def __init__(self, hidden_size: int, hidden_dropout_prob: float, layernorm_eps: float, dtype=None):
        super().__init__(module_name="AttentionOutput")
        dtype = dtype or 'float'
        self.set_dataspec({
            'weight': DataSpec((hidden_size, hidden_size), dtype),
            'bias': DataSpec((hidden_size,), dtype),
            'ln_weight': DataSpec((hidden_size,), dtype),
            'ln_bias': DataSpec((hidden_size,), dtype),
            'hidden': DataSpec((hidden_size,), dtype),
            'input': DataSpec((hidden_size,), dtype),
        })
        self.dropout_prob = hidden_dropout_prob
        self.normalized_shape = (hidden_size,)
        self.ln_eps = layernorm_eps

    def compute(self):
        hidden = self.linear(self.hidden, self.weight, self.bias)
        hidden = self.dropout(hidden, p=self.dropout_prob, train=True)
        hidden = self.layernorm(hidden + self.input, self.normalized_shape, self.ln_weight, self.ln_bias, self.ln_eps)
        return hidden
