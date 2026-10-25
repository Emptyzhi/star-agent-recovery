"""Import-only RPN symbols for ALFWorld's unused visual path."""


class _UnavailableVisualComponent:
    def __init__(self, *_args, **_kwargs):
        raise RuntimeError(
            "visual torchvision components are unavailable in the text-only ALFWorld run"
        )


class AnchorGenerator(_UnavailableVisualComponent):
    pass


class RPNHead(_UnavailableVisualComponent):
    pass
