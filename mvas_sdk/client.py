"""
MVAS Python Client SDK

Simple client for interacting with the MVAS server API.
"""

import base64
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

import requests
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

logger = logging.getLogger(__name__)


@dataclass
class InspectionResult:
    """Result of an inspection"""
    request_id: str
    decision: str
    confidence: float
    anomaly_score: float
    inference_time_ms: float
    total_time_ms: float
    details: Dict[str, Any] = None
    visualization: Optional[np.ndarray] = None
    
    @property
    def is_pass(self) -> bool:
        return self.decision == "pass"
    
    @property
    def is_fail(self) -> bool:
        return self.decision == "fail"


@dataclass
class AppInfo:
    """Application information"""
    id: str
    name: str
    version: str
    description: str
    model_type: str


@dataclass
class CameraInfo:
    """Camera information"""
    id: str
    name: str
    type: str
    address: str
    connected: bool


class MVASClient:
    """
    MVAS Python Client SDK
    
    Example usage:
    ```python
    client = MVASClient("http://localhost:8000")
    
    # Load an application
    app = client.load_app("path/to/app.mvapp")
    
    # Run inspection on an image
    result = client.inspect(app.id, image_path="test.jpg")
    print(f"Decision: {result.decision}")
    
    # Or with numpy array
    import cv2
    image = cv2.imread("test.jpg")
    result = client.inspect(app.id, image=image)
    ```
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize the MVAS client.
        
        Args:
            base_url: MVAS server URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
        
        if api_key:
            self.session.headers["Authorization"] = f"Bearer {api_key}"
    
    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make an API request"""
        url = f"{self.base_url}/api/v1{endpoint}"
        
        kwargs.setdefault("timeout", self.timeout)
        
        response = self.session.request(method, url, **kwargs)
        
        if response.status_code >= 400:
            try:
                error = response.json()
                raise Exception(error.get("detail", response.text))
            except json.JSONDecodeError:
                raise Exception(response.text)
        
        return response.json()
    
    # ========================================================================
    # Health & Status
    # ========================================================================
    
    def health(self) -> bool:
        """Check if server is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def status(self) -> Dict[str, Any]:
        """Get server status"""
        return self._request("GET", "/status")
    
    # ========================================================================
    # Application Management
    # ========================================================================
    
    def load_app(self, app_path: str) -> AppInfo:
        """
        Load an application from a .mvapp file.
        
        Args:
            app_path: Path to the .mvapp file (on the server)
            
        Returns:
            AppInfo with application details
        """
        response = self._request("POST", "/apps/load", json={"app_path": app_path})
        
        info = response.get("app_info", {})
        return AppInfo(
            id=info.get("id"),
            name=info.get("name"),
            version=info.get("version"),
            description=info.get("description", ""),
            model_type=info.get("model_type", ""),
        )
    
    def unload_app(self, app_id: str) -> bool:
        """Unload an application"""
        self._request("DELETE", f"/apps/{app_id}")
        return True
    
    def list_apps(self) -> List[AppInfo]:
        """List all loaded applications"""
        response = self._request("GET", "/apps")
        
        apps = []
        for info in response:
            apps.append(AppInfo(
                id=info.get("id"),
                name=info.get("name"),
                version=info.get("version"),
                description=info.get("description", ""),
                model_type=info.get("model_type", ""),
            ))
        return apps
    
    def get_app(self, app_id: str) -> AppInfo:
        """Get application details"""
        info = self._request("GET", f"/apps/{app_id}")
        return AppInfo(
            id=info.get("id"),
            name=info.get("name"),
            version=info.get("version"),
            description=info.get("description", ""),
            model_type=info.get("model_type", ""),
        )
    
    def get_app_stats(self, app_id: str) -> Dict[str, Any]:
        """Get application statistics"""
        return self._request("GET", f"/apps/{app_id}/stats")
    
    # ========================================================================
    # Inspection
    # ========================================================================
    
    def inspect(
        self,
        app_id: str,
        image: Optional[np.ndarray] = None,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
        visualize: bool = False,
    ) -> InspectionResult:
        """
        Run inspection on an image.
        
        Args:
            app_id: Application ID to use
            image: Image as numpy array (BGR format)
            image_path: Path to image file
            image_base64: Base64 encoded image
            visualize: Return visualization image
            
        Returns:
            InspectionResult with decision and details
        """
        # Prepare image data
        if image is not None:
            if not HAS_CV2:
                raise ImportError("OpenCV required for numpy array input")
            _, buffer = cv2.imencode(".jpg", image)
            image_base64 = base64.b64encode(buffer).decode("utf-8")
        
        elif image_path:
            with open(image_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        if not image_base64:
            raise ValueError("No image provided")
        
        # Make request
        response = self._request("POST", "/inspect", json={
            "app_id": app_id,
            "image_base64": image_base64,
        })
        
        # Build result
        result = InspectionResult(
            request_id=response.get("request_id"),
            decision=response.get("decision"),
            confidence=response.get("confidence"),
            anomaly_score=response.get("anomaly_score"),
            inference_time_ms=response.get("inference_time_ms"),
            total_time_ms=response.get("total_time_ms"),
            details=response.get("details", {}),
        )
        
        # Decode visualization if present
        if visualize and "visualization_base64" in response:
            if HAS_CV2:
                viz_data = base64.b64decode(response["visualization_base64"])
                nparr = np.frombuffer(viz_data, np.uint8)
                result.visualization = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return result
    
    def inspect_file(
        self,
        app_id: str,
        file_path: str,
        visualize: bool = False,
    ) -> InspectionResult:
        """
        Upload and inspect a file.
        
        Args:
            app_id: Application ID
            file_path: Path to image file
            visualize: Return visualization
            
        Returns:
            InspectionResult
        """
        with open(file_path, "rb") as f:
            files = {"file": (Path(file_path).name, f)}
            params = {"app_id": app_id, "visualize": str(visualize).lower()}
            
            response = self.session.post(
                f"{self.base_url}/api/v1/inspect/upload",
                files=files,
                params=params,
                timeout=self.timeout,
            )
        
        if response.status_code >= 400:
            raise Exception(response.text)
        
        data = response.json()
        
        result = InspectionResult(
            request_id=data.get("request_id"),
            decision=data.get("decision"),
            confidence=data.get("confidence"),
            anomaly_score=data.get("anomaly_score"),
            inference_time_ms=data.get("inference_time_ms"),
            total_time_ms=data.get("total_time_ms"),
        )
        
        if visualize and "visualization_base64" in data and HAS_CV2:
            viz_data = base64.b64decode(data["visualization_base64"])
            nparr = np.frombuffer(viz_data, np.uint8)
            result.visualization = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return result
    
    def inspect_batch(
        self,
        app_id: str,
        image_paths: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Run batch inspection on multiple images.
        
        Args:
            app_id: Application ID
            image_paths: List of image file paths
            
        Returns:
            List of result dictionaries
        """
        files = []
        for path in image_paths:
            files.append(("files", (Path(path).name, open(path, "rb"))))
        
        try:
            response = self.session.post(
                f"{self.base_url}/api/v1/inspect/batch",
                files=files,
                params={"app_id": app_id},
                timeout=self.timeout * len(image_paths),
            )
        finally:
            for _, (_, f) in files:
                f.close()
        
        if response.status_code >= 400:
            raise Exception(response.text)
        
        return response.json().get("results", [])
    
    # ========================================================================
    # Camera Management
    # ========================================================================
    
    def connect_camera(
        self,
        camera_type: str,
        address: str,
        name: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> CameraInfo:
        """
        Connect to a camera.
        
        Args:
            camera_type: Type of camera ("usb", "gige", "folder", "rtsp")
            address: Camera address (device ID, IP, path)
            name: Optional friendly name
            settings: Camera settings
            
        Returns:
            CameraInfo
        """
        response = self._request("POST", "/cameras/connect", json={
            "camera_type": camera_type,
            "address": address,
            "name": name,
            "settings": settings or {},
        })
        
        return CameraInfo(
            id=response.get("id"),
            name=response.get("name"),
            type=response.get("type"),
            address=response.get("address"),
            connected=response.get("connected"),
        )
    
    def disconnect_camera(self, camera_id: str) -> bool:
        """Disconnect a camera"""
        self._request("DELETE", f"/cameras/{camera_id}")
        return True
    
    def list_cameras(self) -> List[CameraInfo]:
        """List connected cameras"""
        response = self._request("GET", "/cameras")
        
        cameras = []
        for info in response:
            cameras.append(CameraInfo(
                id=info.get("id"),
                name=info.get("name"),
                type=info.get("type"),
                address=info.get("address"),
                connected=info.get("connected"),
            ))
        return cameras
    
    def grab_frame(self, camera_id: str) -> Optional[np.ndarray]:
        """
        Grab a frame from camera.
        
        Args:
            camera_id: Camera ID
            
        Returns:
            Image as numpy array (BGR)
        """
        if not HAS_CV2:
            raise ImportError("OpenCV required for this method")
        
        response = self.session.get(
            f"{self.base_url}/api/v1/cameras/{camera_id}/grab",
            timeout=self.timeout,
        )
        
        if response.status_code >= 400:
            return None
        
        nparr = np.frombuffer(response.content, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    def inspect_from_camera(
        self,
        app_id: str,
        camera_id: str,
    ) -> InspectionResult:
        """
        Grab frame from camera and run inspection.
        
        Args:
            app_id: Application ID
            camera_id: Camera ID
            
        Returns:
            InspectionResult
        """
        response = self._request(
            "POST",
            f"/cameras/{camera_id}/inspect",
            params={"app_id": app_id},
        )
        
        return InspectionResult(
            request_id=response.get("request_id"),
            decision=response.get("decision"),
            confidence=response.get("confidence"),
            anomaly_score=response.get("anomaly_score"),
            inference_time_ms=response.get("inference_time_ms"),
            total_time_ms=0,
        )
    
    # ========================================================================
    # Streaming (WebSocket)
    # ========================================================================
    
    def stream(
        self,
        app_id: str,
        camera_id: str,
        callback,
        fps: int = 10,
        visualize: bool = True,
    ):
        """
        Start live streaming inspection.
        
        Args:
            app_id: Application ID
            camera_id: Camera ID
            callback: Function to call for each result (receives InspectionResult)
            fps: Target frames per second
            visualize: Include visualization in results
        """
        if not HAS_WEBSOCKET:
            raise ImportError("websocket-client required: pip install websocket-client")
        
        import uuid
        session_id = str(uuid.uuid4())[:8]
        
        ws_url = self.base_url.replace("http", "ws") + f"/ws/stream/{session_id}"
        
        def on_message(ws, message):
            data = json.loads(message)
            
            if data.get("type") == "frame":
                result = InspectionResult(
                    request_id=data.get("timestamp"),
                    decision=data["result"].get("decision"),
                    confidence=data["result"].get("confidence"),
                    anomaly_score=data["result"].get("anomaly_score"),
                    inference_time_ms=data["result"].get("inference_time_ms"),
                    total_time_ms=0,
                )
                
                if "image_base64" in data and HAS_CV2:
                    viz_data = base64.b64decode(data["image_base64"])
                    nparr = np.frombuffer(viz_data, np.uint8)
                    result.visualization = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                callback(result)
        
        def on_open(ws):
            ws.send(json.dumps({
                "type": "config",
                "app_id": app_id,
                "camera_id": camera_id,
                "fps": fps,
                "visualize": visualize,
            }))
        
        ws = websocket.WebSocketApp(
            ws_url,
            on_message=on_message,
            on_open=on_open,
        )
        
        ws.run_forever()

