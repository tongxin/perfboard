import copy
from importlib import import_module
from typing import Any

from common.tensor import DataSpec
from modules import module_classes

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

    def load_ops(self, api):
        for name in dir(api):
            setattr(self, name, getattr(api, name))
        return self

    def prepare_data(self):
        for dataname, spec in self.dataspec.items():
            setattr(self, dataname, self.data_gen(spec))
        return self

    def compute(self):
        pass

class TestModule:
    def __init__(self, module: Module, *args, **kwargs) -> None:
        self.module = module

    def run(self, framework: str, repeat: int):
        global current_framework
        if framework == 'torch':
            from . import torch as api 
        elif framework == 'tfxla':
            from . import tfxla as api
        elif framework == 'jax':
            from . import jax as api
        else:
            raise ValueError()
        
        self.module.load_ops(api).prepare_data()

        for _ in range(repeat):
            self.module.compute()
        


if __name__ == '__main__':
    all_tests = []
    for mod in module_classes:
        if hasattr(mod, 'TESTS'):
            all_tests += [TestModule(mod(*args)) for args in mod.TESTS]

    for test in all_tests:
        test.run('torch', repeat=3)
