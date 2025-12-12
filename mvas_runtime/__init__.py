"""
MVAS - Machine Vision Application Standard
Runtime Package

A plug-and-play framework for deploying machine vision applications.
"""

__version__ = "1.0.0"
__author__ = "MVAS Development Team"

from .config import MVASConfig
from .app_loader import AppLoader, AppInstance
from .app_manager import AppManager
from .inference_engine import InferenceEngine

__all__ = [
    "MVASConfig",
    "AppLoader",
    "AppInstance", 
    "AppManager",
    "InferenceEngine",
]

