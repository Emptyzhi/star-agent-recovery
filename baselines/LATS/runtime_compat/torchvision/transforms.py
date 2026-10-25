"""Small import surface required by ALFWorld's unused visual controller."""

from enum import Enum
import sys
import types


class InterpolationMode(Enum):
    NEAREST = 0
    BOX = 4
    BILINEAR = 2
    HAMMING = 5
    BICUBIC = 3
    LANCZOS = 1


class Compose:
    def __init__(self, transforms):
        self.transforms = tuple(transforms)

    def __call__(self, *args, **kwargs):
        value = args[0] if args else None
        for transform in self.transforms:
            value = transform(value)
        return value


class ToTensor:
    def __call__(self, *_args, **_kwargs):
        raise RuntimeError(
            "torchvision visual transforms are unavailable in the text-only ALFWorld run"
        )


functional = types.ModuleType(__name__ + ".functional")


def _unsupported(*_args, **_kwargs):
    raise RuntimeError(
        "torchvision visual transforms are unavailable in the text-only ALFWorld run"
    )


functional.to_tensor = _unsupported
sys.modules[functional.__name__] = functional

__all__ = ["Compose", "InterpolationMode", "ToTensor", "functional"]
