"""Import-only Mask R-CNN symbols for ALFWorld's unused visual path."""


class MaskRCNNPredictor:
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError(
            "visual torchvision components are unavailable in the text-only ALFWorld run"
        )
