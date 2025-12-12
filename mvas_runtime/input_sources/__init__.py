"""
MVAS Input Sources

Provides unified interface for various image input sources.
"""

from .base import InputSource, InputSourceFactory
from .usb_camera import USBCamera
from .folder_watcher import FolderWatcher
from .rest_upload import RESTUploadSource

__all__ = [
    "InputSource",
    "InputSourceFactory",
    "USBCamera",
    "FolderWatcher",
    "RESTUploadSource",
]

# Try to import GigE camera (requires harvesters library)
try:
    from .gige_camera import GigECamera
    __all__.append("GigECamera")
except ImportError:
    pass

