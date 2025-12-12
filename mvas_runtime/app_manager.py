"""
MVAS Application Manager

Central coordinator for loading, managing, and running applications.
"""

import logging
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np

from .config import get_config
from .app_loader import AppLoader, AppInstance, AppLoadError
from .models import (
    AppInfo, InspectionResult, DecisionResult, InferenceOutput,
    InputSourceType, CameraInfo
)
from .input_sources.base import InputSource, InputSourceFactory, register_default_sources
from .input_sources.rest_upload import RESTUploadSource

logger = logging.getLogger(__name__)


class AppManager:
    """
    Singleton application manager.
    
    Manages:
    - Loading/unloading applications
    - Input source connections
    - Inference execution
    - Result processing
    - Statistics tracking
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.config = get_config()
        self.loader = AppLoader(self.config.storage.temp_dir)
        
        # Loaded applications
        self._apps: Dict[str, AppInstance] = {}
        
        # Connected input sources
        self._sources: Dict[str, InputSource] = {}
        
        # Session tracking
        self._active_sessions: Dict[str, dict] = {}
        
        # Global statistics
        self._stats = {
            "total_inferences": 0,
            "start_time": datetime.now(),
        }
        
        # Register default input sources
        register_default_sources()
        
        # Create default REST upload source
        self._default_rest_source = RESTUploadSource("default_rest", {})
        self._default_rest_source.connect()
        
        self._initialized = True
        logger.info("AppManager initialized")
    
    # ========================================================================
    # Application Management
    # ========================================================================
    
    def load_app(self, app_path: str | Path) -> AppInstance:
        """
        Load an application from a .mvapp package.
        
        Args:
            app_path: Path to the .mvapp file
            
        Returns:
            Loaded AppInstance
            
        Raises:
            AppLoadError: If loading fails
        """
        app_path = Path(app_path)
        
        # Load the application
        app = self.loader.load(app_path)
        
        # Check for duplicate
        if app.app_id in self._apps:
            logger.warning(f"App {app.app_id} already loaded, replacing")
            self.unload_app(app.app_id)
        
        # Register app
        self._apps[app.app_id] = app
        logger.info(f"App loaded: {app.app_id}")
        
        return app
    
    def unload_app(self, app_id: str) -> bool:
        """
        Unload an application.
        
        Args:
            app_id: Application ID to unload
            
        Returns:
            True if unloaded successfully
        """
        if app_id not in self._apps:
            logger.warning(f"App not found: {app_id}")
            return False
        
        app = self._apps.pop(app_id)
        app.cleanup()
        
        logger.info(f"App unloaded: {app_id}")
        return True
    
    def get_app(self, app_id: str) -> Optional[AppInstance]:
        """Get a loaded application by ID"""
        return self._apps.get(app_id)
    
    def list_apps(self) -> list[AppInfo]:
        """List all loaded applications"""
        return [app.info for app in self._apps.values()]
    
    def get_app_stats(self, app_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for an application"""
        app = self._apps.get(app_id)
        if app:
            return app.stats
        return None
    
    # ========================================================================
    # Input Source Management
    # ========================================================================
    
    def connect_camera(
        self,
        camera_type: InputSourceType | str,
        address: str,
        name: Optional[str] = None,
        config: Dict[str, Any] = None,
    ) -> CameraInfo:
        """
        Connect to an input source (camera/folder/etc).
        
        Args:
            camera_type: Type of source
            address: Source address (IP, device ID, path)
            name: Optional friendly name
            config: Additional configuration
            
        Returns:
            CameraInfo with connection details
        """
        # Generate ID
        source_id = name or f"{camera_type}_{len(self._sources)}"
        
        if isinstance(camera_type, InputSourceType):
            camera_type = camera_type.value
        
        # Create source
        source = InputSourceFactory.create(
            camera_type, source_id, address, config or {}
        )
        
        # Connect
        if not source.connect():
            raise RuntimeError(f"Failed to connect to {camera_type} at {address}")
        
        # Register
        self._sources[source_id] = source
        
        return CameraInfo(
            id=source_id,
            name=name or source_id,
            type=InputSourceType(camera_type),
            address=address,
            connected=True,
            settings=config or {},
        )
    
    def disconnect_camera(self, camera_id: str) -> bool:
        """Disconnect an input source"""
        if camera_id not in self._sources:
            return False
        
        source = self._sources.pop(camera_id)
        source.disconnect()
        return True
    
    def get_camera(self, camera_id: str) -> Optional[InputSource]:
        """Get a connected input source"""
        return self._sources.get(camera_id)
    
    def list_cameras(self) -> list[CameraInfo]:
        """List all connected input sources"""
        cameras = []
        for source_id, source in self._sources.items():
            info = source.get_info()
            cameras.append(CameraInfo(
                id=source_id,
                name=source_id,
                type=InputSourceType(info.get("type", "usb")),
                address=info.get("address", info.get("device_id", "")),
                connected=source.is_connected,
                settings=source.config,
            ))
        return cameras
    
    def grab_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """Grab a single frame from a camera"""
        source = self._sources.get(camera_id)
        if source:
            return source.grab_image()
        return None
    
    # ========================================================================
    # Inference
    # ========================================================================
    
    def inspect(
        self,
        app_id: str,
        image: Optional[np.ndarray] = None,
        image_base64: Optional[str] = None,
        image_path: Optional[str] = None,
        camera_id: Optional[str] = None,
    ) -> InspectionResult:
        """
        Run inspection on an image.
        
        Args:
            app_id: Application ID to use
            image: Image as numpy array
            image_base64: Base64 encoded image
            image_path: Path to image file
            camera_id: Grab from camera
            
        Returns:
            InspectionResult with decision and details
        """
        start_time = time.perf_counter()
        request_id = str(uuid.uuid4())[:8]
        
        # Get application
        app = self._apps.get(app_id)
        if app is None:
            return InspectionResult(
                request_id=request_id,
                app_id=app_id,
                decision=DecisionResult.ERROR,
                confidence=0.0,
                anomaly_score=0.0,
                details={"error": f"Application not found: {app_id}"},
                inference_time_ms=0.0,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Get image from various sources
        try:
            if image is not None:
                pass  # Use provided image
            elif image_base64:
                image = RESTUploadSource.decode_base64(image_base64)
            elif image_path:
                import cv2
                image = cv2.imread(image_path)
            elif camera_id:
                source = self._sources.get(camera_id)
                if source:
                    image = source.grab_image()
            
            if image is None:
                raise ValueError("No valid image provided")
                
        except Exception as e:
            return InspectionResult(
                request_id=request_id,
                app_id=app_id,
                decision=DecisionResult.ERROR,
                confidence=0.0,
                anomaly_score=0.0,
                details={"error": f"Failed to get image: {e}"},
                inference_time_ms=0.0,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Run preprocessing
        try:
            input_tensor = app.preprocessing.process(image)
        except Exception as e:
            logger.exception(f"Preprocessing failed: {e}")
            return InspectionResult(
                request_id=request_id,
                app_id=app_id,
                decision=DecisionResult.ERROR,
                confidence=0.0,
                anomaly_score=0.0,
                details={"error": f"Preprocessing failed: {e}"},
                inference_time_ms=0.0,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Run inference
        try:
            output = app.inference_engine.infer(input_tensor)
        except Exception as e:
            logger.exception(f"Inference failed: {e}")
            return InspectionResult(
                request_id=request_id,
                app_id=app_id,
                decision=DecisionResult.ERROR,
                confidence=0.0,
                anomaly_score=0.0,
                details={"error": f"Inference failed: {e}"},
                inference_time_ms=0.0,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
            )
        
        # Apply decision rules
        decision, confidence = self._apply_rules(output, app.rules)
        
        # Calculate times
        total_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Build result
        result = InspectionResult(
            request_id=request_id,
            app_id=app_id,
            decision=decision,
            confidence=confidence,
            anomaly_score=output.anomaly_score,
            details={
                "class_scores": output.class_scores,
                "class_label": output.class_label,
            },
            inference_time_ms=output.inference_time_ms,
            total_time_ms=total_time_ms,
        )
        
        # Update stats
        app.update_stats(decision.value, output.inference_time_ms)
        self._stats["total_inferences"] += 1
        
        logger.debug(f"Inspection complete: {result.decision} ({result.anomaly_score:.3f})")
        
        return result
    
    def _apply_rules(
        self, 
        output: InferenceOutput, 
        rules: Any
    ) -> tuple[DecisionResult, float]:
        """Apply decision rules to inference output"""
        score = output.anomaly_score
        
        # Get thresholds
        thresholds = rules.thresholds.get("anomaly_score", {})
        pass_thresh = thresholds.get("pass", thresholds.get("pass_threshold", 0.3))
        review_thresh = thresholds.get("review", thresholds.get("review_threshold", 0.7))
        
        # Apply simple threshold logic
        if score < pass_thresh:
            return DecisionResult.PASS, 1.0 - score
        elif score < review_thresh:
            return DecisionResult.REVIEW, score
        else:
            return DecisionResult.FAIL, score
    
    # ========================================================================
    # System Status
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        import torch
        
        uptime = (datetime.now() - self._stats["start_time"]).total_seconds()
        
        gpu_available = torch.cuda.is_available() if torch else False
        gpu_name = torch.cuda.get_device_name(0) if gpu_available else None
        
        return {
            "version": "1.0.0",
            "uptime_seconds": uptime,
            "loaded_apps": len(self._apps),
            "connected_cameras": len(self._sources),
            "active_streams": len(self._active_sessions),
            "total_inferences": self._stats["total_inferences"],
            "gpu_available": gpu_available,
            "gpu_name": gpu_name,
        }
    
    def shutdown(self):
        """Shutdown the manager and cleanup resources"""
        logger.info("Shutting down AppManager...")
        
        # Disconnect all cameras
        for source_id in list(self._sources.keys()):
            self.disconnect_camera(source_id)
        
        # Unload all apps
        for app_id in list(self._apps.keys()):
            self.unload_app(app_id)
        
        # Disconnect default REST source
        if self._default_rest_source:
            self._default_rest_source.disconnect()
        
        logger.info("AppManager shutdown complete")


# Convenience function
def get_app_manager() -> AppManager:
    """Get the global AppManager instance"""
    return AppManager()

