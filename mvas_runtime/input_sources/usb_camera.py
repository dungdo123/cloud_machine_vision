"""
MVAS USB Camera Input Source

Supports webcams and USB industrial cameras via OpenCV.
"""

import logging
import time
from typing import Optional, Dict, Any

import cv2
import numpy as np

from .base import InputSource, InputSourceState

logger = logging.getLogger(__name__)


class USBCamera(InputSource):
    """
    USB camera input source using OpenCV VideoCapture.
    
    Supports:
    - Webcams
    - USB industrial cameras
    - DirectShow devices (Windows)
    - V4L2 devices (Linux)
    
    Configuration:
    - address: Device ID (integer) or device path (string)
    - width: Frame width (optional)
    - height: Frame height (optional)
    - fps: Frames per second (optional)
    - exposure: Exposure value (optional)
    - auto_exposure: Enable auto exposure (optional)
    """
    
    def __init__(self, source_id: str, config: Dict[str, Any] = None):
        super().__init__(source_id, config)
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Parse address (device ID)
        address = self.config.get("address", 0)
        if isinstance(address, str):
            try:
                self.device_id = int(address)
            except ValueError:
                self.device_id = address  # Keep as path
        else:
            self.device_id = address
    
    def connect(self) -> bool:
        """Connect to the USB camera"""
        try:
            logger.info(f"Connecting to USB camera: {self.device_id}")
            self.state = InputSourceState.CONNECTING
            
            # Try different backends
            backends = [cv2.CAP_ANY]
            if hasattr(cv2, 'CAP_DSHOW'):
                backends.insert(0, cv2.CAP_DSHOW)  # Windows DirectShow
            
            for backend in backends:
                self.cap = cv2.VideoCapture(self.device_id, backend)
                if self.cap.isOpened():
                    break
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.device_id}")
                self.state = InputSourceState.ERROR
                return False
            
            # Apply settings
            self._apply_settings()
            
            # Grab a test frame
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to grab test frame")
                self.state = InputSourceState.ERROR
                return False
            
            self.state = InputSourceState.CONNECTED
            logger.info(f"USB camera connected: {self.get_info()}")
            return True
            
        except Exception as e:
            logger.exception(f"Error connecting to USB camera: {e}")
            self.state = InputSourceState.ERROR
            return False
    
    def _apply_settings(self):
        """Apply configuration settings to the camera"""
        if self.cap is None:
            return
        
        # Resolution
        if "width" in self.config:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["width"])
        if "height" in self.config:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["height"])
        
        # FPS
        if "fps" in self.config:
            self.cap.set(cv2.CAP_PROP_FPS, self.config["fps"])
        
        # Exposure
        if "auto_exposure" in self.config:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 
                        1 if self.config["auto_exposure"] else 0)
        if "exposure" in self.config:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config["exposure"])
        
        # Gain
        if "gain" in self.config:
            self.cap.set(cv2.CAP_PROP_GAIN, self.config["gain"])
        
        # Autofocus
        if "autofocus" in self.config:
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 
                        1 if self.config["autofocus"] else 0)
    
    def disconnect(self) -> None:
        """Disconnect from the USB camera"""
        self.stop_streaming()
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        
        self.state = InputSourceState.DISCONNECTED
        logger.info(f"USB camera {self.source_id} disconnected")
    
    def grab_image(self) -> Optional[np.ndarray]:
        """Grab a single frame from the camera"""
        if not self.is_connected or self.cap is None:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            logger.warning("Failed to grab frame")
            return None
        
        return frame
    
    def get_info(self) -> Dict[str, Any]:
        """Get camera information"""
        info = {
            "source_id": self.source_id,
            "type": "usb",
            "device_id": self.device_id,
            "state": self.state.value,
        }
        
        if self.cap is not None and self.cap.isOpened():
            info.update({
                "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "backend": self.cap.getBackendName(),
            })
        
        return info
    
    def configure(self, settings: Dict[str, Any]) -> bool:
        """Apply new settings to the camera"""
        self.config.update(settings)
        self._apply_settings()
        return True

