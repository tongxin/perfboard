import copy
from importlib import import_module
from typing import Any

from common.tensor import DataSpec

current_framework = None

class Module:
    """`Module` is the base class for constructing a framework independent test unit
    subjecting to performance measurement. `Module` provides interfaces for mapping
    operators and loading data. A real test module can be defined by extending `Module`
    and overriding its `compute` method.

    NOTE(tongxin): operators differ in semantics and calling convention across different
    frameworks. We try to level off discrepencies by wrapping up operators in case
    the interfaces are not compatible.
    """    
    def __init__(self, module_name: str= None, **kwargs) -> None:
        self.module_name = module_name
        for attr, value in kwargs:
            setattr(self, attr, value)

    def set_dataspec(self, dataspec: dict):
        self.dataspec = dataspec

    def map_ops(self, ops: dict):
        for opname, op in ops.items():
            setattr(self, opname, op)
        return self

    def prepare_data(self, datagen):
        for dataname, spec in self.dataspec.items():
            setattr(self, dataname, datagen(spec))
        return self

    def compute(self):
        pass

class TestModule:
    def __init__(self, module: Module, input_spec: dict) -> None:
        self.module = module
        self.input_spec = input_spec

    def run(self, framework: str, repeat: int):
        global current_framework
        if framework == 'torch':
            from .torch import ops, data_gen
        elif framework == 'tfxla':
            from .tfxla import ops, data_gen
        elif framework == 'jax':
            from .jax import ops, data_gen
        else:
            raise ValueError()
        
        self.module.map_ops().prepare_data(data_gen)

        for _ in range(repeat):
            self.module.compute()
        
