import copy
from importlib import import_module
from typing import Any

from .common.tensor import DataSpec

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

    def load_ops(self, ops):
        for name in ops.oplist:
            setattr(self, name, getattr(ops, name))
        return self

    def prepare_data(self, gen):
        for dataname, spec in self.dataspec.items():
            setattr(self, dataname, gen.data_gen(spec))
        return self

    def compute(self):
        pass

class TestModule:
    def __init__(self, module: Module, *args, **kwargs) -> None:
        self.module = module

    def run(self, framework: str, repeat: int):
        global current_framework

        if framework == 'torch':
            from .torch import _ops as ops
            from .torch import _datagen as gen
        elif framework == 'tfxla':
            pass
        elif framework == 'jax':
            pass
        else:
            raise ValueError()

        current_framework = framework

        self.module.load_ops(ops).prepare_data(gen)

        for _ in range(repeat):
            self.module.compute()

        current_framework = None


