"""Import-only torchvision shim for ALFWorld's text environment.

ALFWorld 0.3.5 imports its visual environment and controller modules while
constructing ``AlfredTWEnv``. The benchmark run uses the text controller, so
the visual API is intentionally unavailable rather than approximated.
"""

from . import models, transforms

__all__ = ["models", "transforms"]
__version__ = "0.0-text-only"
