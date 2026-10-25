"""Explicitly unavailable visual model namespace.

The text-only ALFWorld adapter imports these names through ALFWorld's visual
modules but never constructs them. Raising on construction preserves the
protocol boundary if a visual controller is selected accidentally.
"""

from .faster_rcnn import FastRCNNPredictor
from .mask_rcnn import MaskRCNNPredictor
from .rpn import AnchorGenerator, RPNHead


class _UnavailableVisionModel:
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError(
            "visual torchvision models are unavailable in the text-only ALFWorld run"
        )


class MaskRCNN(_UnavailableVisionModel):
    pass


class KeypointRCNN(_UnavailableVisionModel):
    pass


def maskrcnn_resnet50_fpn(*_args, **_kwargs):
    raise RuntimeError(
        "visual torchvision models are unavailable in the text-only ALFWorld run"
    )


__all__ = [
    "AnchorGenerator",
    "FastRCNNPredictor",
    "KeypointRCNN",
    "MaskRCNN",
    "MaskRCNNPredictor",
    "RPNHead",
    "maskrcnn_resnet50_fpn",
]
