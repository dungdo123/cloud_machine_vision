"""
MVAS GigE Vision Camera Input Source

Supports industrial GigE Vision cameras via GenICam/Harvesters.
"""

import logging
import time
from typing import Optional, Dict, Any

import numpy as np

from .base import InputSource, InputSourceState

logger = logging.getLogger(__name__)


class GigECamera(InputSource):
    """
    GigE Vision camera input source using Harvesters (GenICam).
    
    Supports all GigE Vision compliant cameras including:
    - Basler
    - FLIR/Point Grey
    - Allied Vision
    - IDS
    - And many more
    
    Requirements:
    - harvesters library: pip install harvesters
    - GenTL producer (.cti file) from camera manufacturer
    
    Configuration:
    - address: Camera IP address or serial number
    - cti_file: Path to GenTL producer file (optional, auto-detected)
    - width: Frame width (optional)
    - height: Frame height (optional)
    - exposure_us: Exposure time in microseconds (optional)
    - gain_db: Gain in dB (optional)
    - trigger_mode: "continuous", "software", "hardware" (default: continuous)
    - pixel_format: Pixel format (default: auto)
    """
    
    def __init__(self, source_id: str, config: Dict[str, Any] = None):
        super().__init__(source_id, config)
        
        self.address = self.config.get("address", "")
        self.cti_file = self.config.get("cti_file")
        
        self._harvester = None
        self._image_acquirer = None
    
    def connect(self) -> bool:
        """Connect to the GigE camera"""
        try:
            from harvesters.core import Harvester
        except ImportError:
            logger.error("harvesters library not installed. Install with: pip install harvesters")
            self.state = InputSourceState.ERROR
            return False
        
        try:
            logger.info(f"Connecting to GigE camera: {self.address}")
            self.state = InputSourceState.CONNECTING
            
            # Create Harvester instance
            self._harvester = Harvester()
            
            # Load GenTL producer
            cti_file = self._find_cti_file()
            if cti_file is None:
                logger.error("No GenTL producer (.cti) file found")
                self.state = InputSourceState.ERROR
                return False
            
            self._harvester.add_file(cti_file)
            self._harvester.update()
            
            # Find camera by address
            device_info = self._find_camera()
            if device_info is None:
                logger.error(f"Camera not found: {self.address}")
                self.state = InputSourceState.ERROR
                return False
            
            # Create image acquirer
            self._image_acquirer = self._harvester.create(device_info)
            
            # Apply settings
            self._apply_settings()
            
            # Start acquisition
            self._image_acquirer.start()
            
            self.state = InputSourceState.CONNECTED
            logger.info(f"GigE camera connected: {self.get_info()}")
            return True
            
        except Exception as e:
            logger.exception(f"Error connecting to GigE camera: {e}")
            self.state = InputSourceState.ERROR
            return False
    
    def _find_cti_file(self) -> Optional[str]:
        """Find GenTL producer file"""
        import os
        from pathlib import Path
        
        if self.cti_file and os.path.exists(self.cti_file):
            return self.cti_file
        
        # Common locations for CTI files
        search_paths = [
            # Windows
            Path(os.environ.get("GENICAM_GENTL64_PATH", "")),
            Path(os.environ.get("GENICAM_GENTL32_PATH", "")),
            Path("C:/Program Files/MATRIX VISION/mvIMPACT Acquire/bin/x64"),
            Path("C:/Program Files/Basler/pylon 6/Runtime/x64"),
            Path("C:/Program Files/Allied Vision/Vimba_5.0/VimbaGigETL/Bin/Win64"),
            # Linux
            Path("/opt/mvIMPACT_Acquire/lib/x86_64"),
            Path("/opt/pylon/lib"),
        ]
        
        for path in search_paths:
            if not path.exists():
                continue
            for cti in path.glob("*.cti"):
                logger.info(f"Found GenTL producer: {cti}")
                return str(cti)
        
        return None
    
    def _find_camera(self):
        """Find camera by address or serial number"""
        if not self._harvester:
            return None
        
        for device_info in self._harvester.device_info_list:
            # Try to match by various identifiers
            identifiers = [
                device_info.id_,
                device_info.serial_number if hasattr(device_info, 'serial_number') else None,
                device_info.user_defined_name if hasattr(device_info, 'user_defined_name') else None,
            ]
            
            if self.address in [str(i) for i in identifiers if i]:
                return device_info
        
        # If no address specified, return first camera
        if not self.address and self._harvester.device_info_list:
            return self._harvester.device_info_list[0]
        
        return None
    
    def _apply_settings(self):
        """Apply configuration settings to the camera"""
        if not self._image_acquirer:
            return
        
        node_map = self._image_acquirer.remote_device.node_map
        
        try:
            # Exposure
            if "exposure_us" in self.config:
                if hasattr(node_map, "ExposureTime"):
                    node_map.ExposureTime.value = self.config["exposure_us"]
            
            # Gain
            if "gain_db" in self.config:
                if hasattr(node_map, "Gain"):
                    node_map.Gain.value = self.config["gain_db"]
            
            # Resolution
            if "width" in self.config and hasattr(node_map, "Width"):
                node_map.Width.value = self.config["width"]
            if "height" in self.config and hasattr(node_map, "Height"):
                node_map.Height.value = self.config["height"]
            
            # Trigger mode
            trigger_mode = self.config.get("trigger_mode", "continuous")
            if hasattr(node_map, "TriggerMode"):
                if trigger_mode == "continuous":
                    node_map.TriggerMode.value = "Off"
                elif trigger_mode == "software":
                    node_map.TriggerMode.value = "On"
                    node_map.TriggerSource.value = "Software"
                elif trigger_mode == "hardware":
                    node_map.TriggerMode.value = "On"
                    node_map.TriggerSource.value = "Line1"
            
            # Pixel format
            if "pixel_format" in self.config and hasattr(node_map, "PixelFormat"):
                node_map.PixelFormat.value = self.config["pixel_format"]
                
        except Exception as e:
            logger.warning(f"Error applying camera settings: {e}")
    
    def disconnect(self) -> None:
        """Disconnect from the GigE camera"""
        self.stop_streaming()
        
        if self._image_acquirer:
            try:
                self._image_acquirer.stop()
                self._image_acquirer.destroy()
            except Exception as e:
                logger.warning(f"Error stopping image acquirer: {e}")
            self._image_acquirer = None
        
        if self._harvester:
            self._harvester.reset()
            self._harvester = None
        
        self.state = InputSourceState.DISCONNECTED
        logger.info(f"GigE camera {self.source_id} disconnected")
    
    def grab_image(self) -> Optional[np.ndarray]:
        """Grab a single frame from the camera"""
        if not self.is_connected or not self._image_acquirer:
            return None
        
        try:
            # Trigger if in software trigger mode
            trigger_mode = self.config.get("trigger_mode", "continuous")
            if trigger_mode == "software":
                node_map = self._image_acquirer.remote_device.node_map
                if hasattr(node_map, "TriggerSoftware"):
                    node_map.TriggerSoftware.execute()
            
            # Fetch buffer
            timeout = self.config.get("timeout_ms", 5000) / 1000.0
            with self._image_acquirer.fetch(timeout=timeout) as buffer:
                # Get image data
                component = buffer.payload.components[0]
                
                # Convert to numpy array
                image = component.data.reshape(
                    component.height,
                    component.width,
                    -1 if component.num_components_per_pixel > 1 else 1
                )
                
                # Convert to BGR if needed
                if component.num_components_per_pixel == 1:
                    # Mono to BGR
                    image = np.repeat(image, 3, axis=2)
                elif component.data_format == "RGB8":
                    # RGB to BGR
                    image = image[:, :, ::-1]
                
                return image.copy()
                
        except Exception as e:
            logger.error(f"Error grabbing frame: {e}")
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get camera information"""
        info = {
            "source_id": self.source_id,
            "type": "gige",
            "address": self.address,
            "state": self.state.value,
        }
        
        if self._image_acquirer:
            try:
                node_map = self._image_acquirer.remote_device.node_map
                info.update({
                    "width": node_map.Width.value if hasattr(node_map, "Width") else None,
                    "height": node_map.Height.value if hasattr(node_map, "Height") else None,
                    "pixel_format": node_map.PixelFormat.value if hasattr(node_map, "PixelFormat") else None,
                    "model": node_map.DeviceModelName.value if hasattr(node_map, "DeviceModelName") else None,
                })
            except Exception:
                pass
        
        return info

