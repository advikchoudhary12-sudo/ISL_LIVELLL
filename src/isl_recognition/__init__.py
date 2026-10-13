"""From-scratch ISL alphabet recognition."""

import os
import sys

os.environ.setdefault("MEDIAPIPE_DISABLE_GPU", "1")

if sys.platform.startswith("linux"):
    try:
        import ctypes

        ctypes.CDLL("libX11.so.6").XInitThreads()
    except OSError:
        pass

__version__ = "0.1.0"
