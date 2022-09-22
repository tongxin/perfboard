import math
from typing import Optional, Tuple

import torch
from torch import nn

from common.tensor import DataSpec
from benchmark import Module

class AttentionOutput(Module):
    def __init__(self, hidden_size: int, hidden_dropout_prob: float, layernorm_eps: float, dtype=None):
        super().__init__(module_name="AttentionOutput")
        dtype = dtype or 'float'
        self.set_dataspec({
            'weight': DataSpec((hidden_size, hidden_size), dtype),
            'bias' : DataSpec((hidden_size,), dtype),
            'ln_weight': DataSpec((hidden_size, hidden_size), dtype),
            'ln_bias': DataSpec((hidden_size,), dtype),
            'hidden': DataSpec((hidden_size,), dtype),
            'input': DataSpec((hidden_size,), dtype),
        })
        self.dropout_prob = hidden_dropout_prob
        self.ln_eps = layernorm_eps

    def compute(self):
        hidden = self.linear(self.hidden, self.weight, self.bias)
        hidden = self.dropout(hidden, p=self.dropout_prob)
        hidden = self.layernorm(hidden + self.input, self.ln_weight, self.ln_bias, self.ln_eps)
        return hidden
