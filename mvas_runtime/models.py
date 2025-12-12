"""
MVAS Data Models

Pydantic models for all data structures used in the MVAS runtime.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List, Tuple
from pydantic import BaseModel, Field
import numpy as np


# ============================================================================
# Enums
# ============================================================================

class DecisionResult(str, Enum):
    """Inspection decision results"""
    PASS = "pass"
    FAIL = "fail"
    REVIEW = "review"
    ERROR = "error"


class ModelFramework(str, Enum):
    """Supported model frameworks"""
    ONNX = "onnx"
    TORCHSCRIPT = "torchscript"
    TENSORRT = "tensorrt"
    OPENVINO = "openvino"


class ModelType(str, Enum):
    """Types of machine vision models"""
    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"


class InputSourceType(str, Enum):
    """Types of input sources"""
    GIGE_CAMERA = "gige"
    USB_CAMERA = "usb"
    RTSP_STREAM = "rtsp"
    FOLDER = "folder"
    REST_UPLOAD = "rest"


class DeviceType(str, Enum):
    """Compute device types"""
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


# ============================================================================
# Manifest Models (for .mvapp packages)
# ============================================================================

class AppMetadata(BaseModel):
    """Application metadata from manifest"""
    id: str
    name: str
    version: str
    description: str = ""
    author: str = ""
    created: Optional[datetime] = None
    tags: List[str] = []


class ModelConfig(BaseModel):
    """Model configuration from manifest"""
    type: ModelType
    algorithm: str = ""
    framework: ModelFramework
    path: str
    additional_files: Dict[str, str] = {}
    runtime: Dict[str, Any] = {}


class InputConfig(BaseModel):
    """Input configuration from manifest"""
    type: str = "image"
    color_mode: str = "RGB"
    resolution: Dict[str, int] = {"width": 256, "height": 256}
    preprocessing: str = "preprocessing/transforms.json"
    roi_mask: Optional[str] = None


class OutputConfig(BaseModel):
    """Output configuration from manifest"""
    type: str = "anomaly_map"
    postprocessing: str = "postprocessing/rules.json"
    save_images: bool = True
    save_visualizations: bool = True


class RequirementsConfig(BaseModel):
    """Requirements configuration from manifest"""
    min_mvas_version: str = "1.0.0"
    gpu_required: bool = False
    min_memory_mb: int = 256


class Manifest(BaseModel):
    """Complete .mvapp manifest structure"""
    mvas_version: str = "1.0.0"
    app: AppMetadata
    model: ModelConfig
    input: InputConfig
    output: OutputConfig
    requirements: RequirementsConfig = Field(default_factory=RequirementsConfig)


# ============================================================================
# Transform Models
# ============================================================================

class TransformOp(BaseModel):
    """Single transform operation"""
    op: str
    params: Dict[str, Any] = {}
    enabled: bool = True


class TransformPipeline(BaseModel):
    """Complete preprocessing pipeline"""
    pipeline: List[TransformOp]


# ============================================================================
# Rules Models
# ============================================================================

class ThresholdConfig(BaseModel):
    """Threshold configuration for decisions"""
    pass_threshold: float = Field(alias="pass", default=0.3)
    review_threshold: float = Field(alias="review", default=0.7)
    
    class Config:
        populate_by_name = True


class DecisionRule(BaseModel):
    """Single decision rule"""
    condition: str
    decision: DecisionResult
    confidence_source: str = "anomaly_score"


class DecisionLogic(BaseModel):
    """Decision logic configuration"""
    primary: str = "anomaly_score"
    rules: List[DecisionRule]


class RulesConfig(BaseModel):
    """Complete postprocessing rules configuration"""
    thresholds: Dict[str, ThresholdConfig] = {}
    decision_logic: Optional[DecisionLogic] = None
    outputs: Dict[str, bool] = {}
    visualization: Dict[str, Any] = {}


# ============================================================================
# Runtime Models
# ============================================================================

class AppInfo(BaseModel):
    """Application info for API responses"""
    id: str
    name: str
    version: str
    description: str
    model_type: str
    algorithm: str
    input_resolution: Tuple[int, int]
    loaded_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        from_attributes = True


class InferenceOutput(BaseModel):
    """Raw inference output"""
    anomaly_score: float = 0.0
    anomaly_map: Optional[Any] = None  # np.ndarray, but Any for Pydantic
    class_scores: Optional[Dict[str, float]] = None
    class_label: Optional[str] = None
    detections: Optional[List[Dict[str, Any]]] = None
    inference_time_ms: float = 0.0
    
    class Config:
        arbitrary_types_allowed = True


class InspectionResult(BaseModel):
    """Final inspection result"""
    request_id: str
    app_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    decision: DecisionResult
    confidence: float
    anomaly_score: float
    details: Dict[str, Any] = {}
    image_path: Optional[str] = None
    visualization_path: Optional[str] = None
    inference_time_ms: float
    total_time_ms: float
    
    class Config:
        from_attributes = True


# ============================================================================
# API Request/Response Models
# ============================================================================

class LoadAppRequest(BaseModel):
    """Request to load an application"""
    app_path: str


class LoadAppResponse(BaseModel):
    """Response after loading an application"""
    success: bool
    app_id: Optional[str] = None
    app_info: Optional[AppInfo] = None
    message: str


class InspectRequest(BaseModel):
    """Request to run inspection"""
    app_id: str
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    image_path: Optional[str] = None


class InspectResponse(BaseModel):
    """Response from inspection"""
    request_id: str
    decision: DecisionResult
    confidence: float
    anomaly_score: float
    inference_time_ms: float
    total_time_ms: float
    details: Dict[str, Any] = {}
    visualization_base64: Optional[str] = None


class ConnectCameraRequest(BaseModel):
    """Request to connect to a camera"""
    camera_type: InputSourceType
    address: str
    name: Optional[str] = None
    settings: Dict[str, Any] = {}


class CameraInfo(BaseModel):
    """Camera information"""
    id: str
    name: str
    type: InputSourceType
    address: str
    connected: bool
    settings: Dict[str, Any] = {}


class StreamConfig(BaseModel):
    """Streaming configuration"""
    app_id: str
    camera_id: str
    fps: int = 30
    visualize: bool = True


class SystemStatus(BaseModel):
    """System status information"""
    version: str
    uptime_seconds: float
    loaded_apps: int
    connected_cameras: int
    active_streams: int
    gpu_available: bool
    gpu_name: Optional[str] = None

