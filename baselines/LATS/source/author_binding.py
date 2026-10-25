"""Isolate author function globals per concurrent task without rewriting bodies."""
from types import FunctionType, ModuleType


def bind_module(module, **overrides):
    bound = ModuleType(module.__name__)
    namespace = vars(bound)
    namespace.update(vars(module))

    def clone_function(value, class_cell=None):
        closure = value.__closure__
        if '__class__' in value.__code__.co_freevars:
            if class_cell is None:
                raise ValueError('class method requires its new class cell')
            closure = tuple(class_cell if name == '__class__' else cell
                            for name, cell in zip(value.__code__.co_freevars, closure))
        clone = FunctionType(value.__code__, namespace, value.__name__, value.__defaults__, closure)
        clone.__kwdefaults__ = value.__kwdefaults__
        clone.__annotations__ = value.__annotations__
        return clone

    def clone_member(value, class_cell=None):
        if isinstance(value, FunctionType) and value.__globals__ is vars(module):
            return clone_function(value, class_cell)
        if isinstance(value, staticmethod):return staticmethod(clone_member(value.__func__, class_cell))
        if isinstance(value, classmethod):return classmethod(clone_member(value.__func__, class_cell))
        return value

    for name, value in list(namespace.items()):
        if isinstance(value,type) and value.__module__==module.__name__:
            # Zero-argument super() must refer to the cloned class, not its source.
            class_cell = (lambda x: lambda: x)(None).__closure__[0]
            members={k:clone_member(v, class_cell) for k,v in vars(value).items() if k not in ('__dict__','__weakref__')}
            cloned_class=type(value)(value.__name__,value.__bases__,members)
            class_cell.cell_contents=cloned_class
            namespace[name]=cloned_class
        else:
            namespace[name]=clone_member(value)
    namespace.update(overrides)
    return bound
