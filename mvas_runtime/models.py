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
    NATIVE_PLUGIN = "native_plugin"  # Native DLL/SO plugin


class PackageType(str, Enum):
    """Types of .mvapp packages"""
    SCRIPT = "script"           # Python/ONNX based (v1)
    NATIVE = "native"           # Native DLL/SO plugin (v2)


class PluginState(str, Enum):
    """Plugin lifecycle states"""
    LOADING = "loading"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"


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
    cuda_version: Optional[str] = None
    gpu_compute_capability: Optional[str] = None


# ============================================================================
# Native Plugin Manifest Models (v2)
# ============================================================================

class PluginBinaries(BaseModel):
    """Platform-specific plugin binaries"""
    windows_x64: Optional[str] = Field(None, alias="windows-x64")
    linux_x64: Optional[str] = Field(None, alias="linux-x64")
    linux_arm64: Optional[str] = Field(None, alias="linux-arm64")
    macos_x64: Optional[str] = Field(None, alias="macos-x64")
    macos_arm64: Optional[str] = Field(None, alias="macos-arm64")
    
    model_config = {"populate_by_name": True}
    
    def get_binary_for_platform(self, platform_key: str) -> Optional[str]:
        """Get binary path for given platform key"""
        mapping = {
            "windows-x64": self.windows_x64,
            "linux-x64": self.linux_x64,
            "linux-arm64": self.linux_arm64,
            "macos-x64": self.macos_x64,
            "macos-arm64": self.macos_arm64,
        }
        return mapping.get(platform_key)


class PluginModelConfig(BaseModel):
    """Model configuration for native plugins"""
    path: str = "models/model.engine"
    fallback: Optional[str] = "models/model.onnx"
    config: Optional[str] = "models/model_config.json"


class PluginConfig(BaseModel):
    """Native plugin configuration from manifest"""
    interface_version: str = "2.0"
    binaries: PluginBinaries
    model: PluginModelConfig = Field(default_factory=PluginModelConfig)
    dependencies: Dict[str, List[str]] = {}


class PluginInterfaceConfig(BaseModel):
    """Plugin interface configuration"""
    input: InputConfig = Field(default_factory=InputConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


class ConfigurableParam(BaseModel):
    """User-configurable parameter definition"""
    key: str
    type: str = "float"  # float, int, string, enum, bool
    default: Any = None
    min: Optional[float] = None
    max: Optional[float] = None
    options: Optional[List[str]] = None
    label: str = ""
    description: str = ""


class PluginConfiguration(BaseModel):
    """Plugin configuration options"""
    default_config: Optional[str] = None
    user_configurable: List[ConfigurableParam] = []


class Manifest(BaseModel):
    """Complete .mvapp manifest structure (supports both v1 and v2)"""
    mvas_version: str = "1.0.0"
    manifest_version: str = "1.0.0"
    package_type: PackageType = PackageType.SCRIPT
    
    app: AppMetadata
    
    # v1 (script-based) fields
    model: Optional[ModelConfig] = None
    input: Optional[InputConfig] = None
    output: Optional[OutputConfig] = None
    
    # v2 (native plugin) fields
    plugin: Optional[PluginConfig] = None
    interface: Optional[PluginInterfaceConfig] = None
    configuration: Optional[PluginConfiguration] = None
    
    # Common fields
    requirements: RequirementsConfig = Field(default_factory=RequirementsConfig)
    
    def is_native_plugin(self) -> bool:
        """Check if this is a native plugin package"""
        return self.package_type == PackageType.NATIVE or self.plugin is not None


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
    package_type: PackageType = PackageType.SCRIPT
    
    class Config:
        from_attributes = True


class PluginAppInfo(BaseModel):
    """Plugin application info for API responses"""
    id: str
    name: str
    version: str
    description: str
    author: str = ""
    model_type: str
    input_width: int
    input_height: int
    input_format: str = "RGB"
    package_type: PackageType = PackageType.NATIVE
    state: PluginState = PluginState.READY
    load_time_ms: float = 0.0
    avg_inference_ms: float = 0.0
    total_inferences: int = 0
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

