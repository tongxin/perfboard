from ..benchmark import Module, TestModule
from .attention import AttentionOutput


def get_module_classes():
    from inspect import isclass
    return [v for k, v in globals().items() if isclass(v) and issubclass(v, Module)]

module_classes = get_module_classes()
