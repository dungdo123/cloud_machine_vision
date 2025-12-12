"""
MVAS REST Upload Input Source

Receives images via REST API upload.
"""

import base64
import io
import logging
from typing import Optional, Dict, Any
from queue import Queue, Empty

import cv2
import numpy as np

from .base import InputSource, InputSourceState

logger = logging.getLogger(__name__)


class RESTUploadSource(InputSource):
    """
    REST upload input source.
    
    Receives images via REST API calls. This source is typically
    used in conjunction with the MVAS server's /inspect endpoint.
    
    Images can be provided as:
    - Base64 encoded data
    - Binary data
    - URL reference
    
    Configuration:
    - address: Not used (for API compatibility)
    - max_queue_size: Maximum number of images to queue (default: 100)
    """
    
    def __init__(self, source_id: str, config: Dict[str, Any] = None):
        super().__init__(source_id, config)
        
        self.max_queue_size = self.config.get("max_queue_size", 100)
        self._image_queue: Queue = Queue(maxsize=self.max_queue_size)
    
    def connect(self) -> bool:
        """Initialize the REST upload source"""
        self.state = InputSourceState.CONNECTED
        logger.info(f"REST upload source {self.source_id} ready")
        return True
    
    def disconnect(self) -> None:
        """Disconnect the REST upload source"""
        self.stop_streaming()
        
        # Clear queue
        while not self._image_queue.empty():
            try:
                self._image_queue.get_nowait()
            except Empty:
                break
        
        self.state = InputSourceState.DISCONNECTED
        logger.info(f"REST upload source {self.source_id} disconnected")
    
    def grab_image(self) -> Optional[np.ndarray]:
        """Get the next image from the queue"""
        if not self.is_connected:
            return None
        
        try:
            return self._image_queue.get(timeout=1.0)
        except Empty:
            return None
    
    def submit_image(self, image: np.ndarray) -> bool:
        """
        Submit an image to the queue.
        
        Args:
            image: Image as numpy array
            
        Returns:
            True if image was queued successfully
        """
        if not self.is_connected:
            return False
        
        try:
            self._image_queue.put_nowait(image)
            return True
        except:
            logger.warning("Image queue is full, dropping image")
            return False
    
    def submit_base64(self, base64_data: str) -> bool:
        """
        Submit a base64-encoded image.
        
        Args:
            base64_data: Base64 encoded image data
            
        Returns:
            True if image was decoded and queued successfully
        """
        try:
            image = self.decode_base64(base64_data)
            if image is not None:
                return self.submit_image(image)
            return False
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return False
    
    def submit_bytes(self, image_bytes: bytes) -> bool:
        """
        Submit raw image bytes.
        
        Args:
            image_bytes: Raw image data (JPEG, PNG, etc.)
            
        Returns:
            True if image was decoded and queued successfully
        """
        try:
            image = self.decode_bytes(image_bytes)
            if image is not None:
                return self.submit_image(image)
            return False
        except Exception as e:
            logger.error(f"Failed to decode image bytes: {e}")
            return False
    
    @staticmethod
    def decode_base64(base64_data: str) -> Optional[np.ndarray]:
        """
        Decode a base64-encoded image.
        
        Args:
            base64_data: Base64 encoded string (with or without data URI prefix)
            
        Returns:
            Image as numpy array, or None if decoding failed
        """
        try:
            # Remove data URI prefix if present
            if "," in base64_data:
                base64_data = base64_data.split(",", 1)[1]
            
            # Decode base64
            image_bytes = base64.b64decode(base64_data)
            
            # Decode image
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return image
            
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return None
    
    @staticmethod
    def decode_bytes(image_bytes: bytes) -> Optional[np.ndarray]:
        """
        Decode raw image bytes.
        
        Args:
            image_bytes: Raw image data
            
        Returns:
            Image as numpy array, or None if decoding failed
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            logger.error(f"Failed to decode image bytes: {e}")
            return None
    
    @staticmethod
    def encode_base64(image: np.ndarray, format: str = "jpeg") -> str:
        """
        Encode image to base64.
        
        Args:
            image: Image as numpy array
            format: Output format ("jpeg" or "png")
            
        Returns:
            Base64 encoded string
        """
        if format.lower() == "jpeg":
            ext = ".jpg"
            params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        else:
            ext = ".png"
            params = []
        
        _, buffer = cv2.imencode(ext, image, params)
        return base64.b64encode(buffer).decode("utf-8")
    
    def get_queue_size(self) -> int:
        """Get current queue size"""
        return self._image_queue.qsize()
    
    def get_info(self) -> Dict[str, Any]:
        """Get source information"""
        return {
            "source_id": self.source_id,
            "type": "rest",
            "state": self.state.value,
            "queue_size": self.get_queue_size(),
            "max_queue_size": self.max_queue_size,
        }

