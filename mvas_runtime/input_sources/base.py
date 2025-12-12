"""
MVAS Input Source Base Classes

Abstract base class and factory for all input sources.
"""

import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any, Type
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class InputSourceState(Enum):
    """Input source connection state"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"


class InputSource(ABC):
    """
    Abstract base class for all input sources.
    
    Input sources provide images to the MVAS runtime from various
    origins: cameras, folders, network streams, etc.
    """
    
    def __init__(self, source_id: str, config: Dict[str, Any] = None):
        """
        Initialize input source.
        
        Args:
            source_id: Unique identifier for this source
            config: Source-specific configuration
        """
        self.source_id = source_id
        self.config = config or {}
        self.state = InputSourceState.DISCONNECTED
        self._streaming = False
        self._callback: Optional[Callable[[np.ndarray], None]] = None
    
    @property
    def is_connected(self) -> bool:
        """Check if source is connected"""
        return self.state in (InputSourceState.CONNECTED, InputSourceState.STREAMING)
    
    @property
    def is_streaming(self) -> bool:
        """Check if source is actively streaming"""
        return self.state == InputSourceState.STREAMING
    
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the input source.
        
        Returns:
            True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the input source"""
        pass
    
    @abstractmethod
    def grab_image(self) -> Optional[np.ndarray]:
        """
        Grab a single image from the source.
        
        Returns:
            Image as numpy array (HWC format, BGR or RGB), or None if failed
        """
        pass
    
    def start_streaming(self, callback: Callable[[np.ndarray], None]) -> bool:
        """
        Start continuous image streaming.
        
        Args:
            callback: Function to call for each captured image
            
        Returns:
            True if streaming started successfully
        """
        if not self.is_connected:
            logger.error("Cannot start streaming: source not connected")
            return False
        
        self._callback = callback
        self._streaming = True
        self.state = InputSourceState.STREAMING
        
        # Default implementation - subclasses may override for async streaming
        self._start_stream_loop()
        return True
    
    def _start_stream_loop(self):
        """Default streaming loop - override for custom implementation"""
        import threading
        
        def stream_thread():
            while self._streaming:
                image = self.grab_image()
                if image is not None and self._callback:
                    try:
                        self._callback(image)
                    except Exception as e:
                        logger.error(f"Stream callback error: {e}")
        
        self._stream_thread = threading.Thread(target=stream_thread, daemon=True)
        self._stream_thread.start()
    
    def stop_streaming(self) -> None:
        """Stop continuous streaming"""
        self._streaming = False
        self._callback = None
        if self.state == InputSourceState.STREAMING:
            self.state = InputSourceState.CONNECTED
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about the input source.
        
        Returns:
            Dictionary with source information
        """
        pass
    
    def configure(self, settings: Dict[str, Any]) -> bool:
        """
        Configure the input source with new settings.
        
        Args:
            settings: New settings to apply
            
        Returns:
            True if configuration applied successfully
        """
        self.config.update(settings)
        return True


class InputSourceFactory:
    """
    Factory for creating input sources.
    
    Automatically selects the appropriate input source class
    based on the source type.
    """
    
    _sources: Dict[str, Type[InputSource]] = {}
    
    @classmethod
    def register(cls, source_type: str, source_class: Type[InputSource]):
        """Register a new input source type"""
        cls._sources[source_type] = source_class
    
    @classmethod
    def create(
        cls,
        source_type: str,
        source_id: str,
        address: str,
        config: Dict[str, Any] = None
    ) -> InputSource:
        """
        Create an input source.
        
        Args:
            source_type: Type of source ("usb", "gige", "folder", "rtsp")
            source_id: Unique identifier
            address: Source address (device ID, IP, path, etc.)
            config: Additional configuration
            
        Returns:
            InputSource instance
        """
        config = config or {}
        config["address"] = address
        
        if source_type not in cls._sources:
            raise ValueError(f"Unknown source type: {source_type}")
        
        source_class = cls._sources[source_type]
        return source_class(source_id, config)
    
    @classmethod
    def list_types(cls) -> list:
        """List available source types"""
        return list(cls._sources.keys())


# Register source types (actual imports done in __init__.py)
def register_default_sources():
    """Register default input source types"""
    from .usb_camera import USBCamera
    from .folder_watcher import FolderWatcher
    from .rest_upload import RESTUploadSource
    
    InputSourceFactory.register("usb", USBCamera)
    InputSourceFactory.register("folder", FolderWatcher)
    InputSourceFactory.register("rest", RESTUploadSource)
    
    # Try to register GigE camera
    try:
        from .gige_camera import GigECamera
        InputSourceFactory.register("gige", GigECamera)
    except ImportError:
        logger.debug("GigE camera support not available (harvesters not installed)")

