from benchmark import Module, TestModule
from .attention import AttentionOutput

@property
def module_classes():
    return [v for k, v in globals().items() if issubclass(v, Module)]
