import copy
from importlib import import_module
from typing import Any

class Module:
    
    def __init__(self, module_name: str= None, **kwargs) -> None:
        self.module_name = module_name
        for attr, value in kwargs:
            setattr(self, attr, value)

    def load_ops(self, ops: dict):
        for opname, op in ops.items():
            setattr(self, opname, op)
        return self

    def load_data(self, inputs: dict):
        for dataname, data in inputs.items():
            setattr(self, dataname, data)
        return self

    def compute(self):
        pass

class TestFormula:
    def __init__(self, module: Module, input_spec: dict) -> None:
        self.module = module
        self.input_spec = input_spec

    def run(self, framework: str, repeat: int):
        if framework == 'torch':
            from .torch import ops, prepare_data
        elif framework == 'tfxla':
            from .tfxla import ops, prepare_data
        elif framework == 'jax':
            from .jax import ops, prepare_data
        else:
            raise ValueError()
        inputs = prepare_data(self.input_spec)
        self.module.load_ops().load_data(inputs)

        for _ in range(repeat):
            self.module.compute()
        
