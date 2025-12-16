# MVAS - Machine Vision Application Standard
## System Design Document

**Version:** 1.0.0  
**Date:** December 12, 2025  
**Author:** MVAS Development Team

> ⚠️ **Note:** This is the v1.0 basic architecture. For production deployments requiring 
> multi-stage pipelines, multiple models, and app marketplace functionality, see 
> [ENHANCED_ARCHITECTURE.md](./ENHANCED_ARCHITECTURE.md) which describes the v2.0 
> DAG-based pipeline system and App Store distribution model.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Architecture](#3-architecture)
4. [Component Design](#4-component-design)
5. [Data Models](#5-data-models)
6. [API Specification](#6-api-specification)
7. [Application Package Format](#7-application-package-format)
8. [Input Sources](#8-input-sources)
9. [Inference Pipeline](#9-inference-pipeline)
10. [Output Handlers](#10-output-handlers)
11. [Deployment](#11-deployment)
12. [Security](#12-security)
13. [Performance](#13-performance)
14. [Future Roadmap](#14-future-roadmap)

---

## 1. Executive Summary

### 1.1 Purpose

MVAS (Machine Vision Application Standard) is a plug-and-play framework for deploying machine vision applications in industrial environments. It standardizes how vision applications are packaged, deployed, and executed, allowing users to:

- **Select** a pre-built application file (`.mvapp`)
- **Connect** to any supported image source (cameras, folders, streams)
- **Run** inspections immediately without coding

### 1.2 Goals

| Goal | Description |
|------|-------------|
| **Simplicity** | Zero-code deployment for end users |
| **Standardization** | Common format for all vision applications |
| **Flexibility** | Support multiple models, cameras, and outputs |
| **Scalability** | Handle single camera to multi-line factories |
| **Reliability** | Industrial-grade uptime and error handling |

### 1.3 Key Features

- 📦 **Portable Application Packages** - Self-contained `.mvapp` files
- 🔌 **Universal Camera Support** - GigE Vision, USB, RTSP, REST
- ⚡ **Multiple Inference Backends** - ONNX, TorchScript, TensorRT, OpenVINO
- 🌐 **REST & WebSocket APIs** - Easy integration with existing systems
- 📊 **Real-time Monitoring** - Live dashboards and metrics
- 🔄 **Hot-swap Applications** - Change models without restart

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACE                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │  Web UI     │  │  REST API   │  │  CLI Tool   │  │  Python SDK     │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘    │
└─────────┼────────────────┼────────────────┼──────────────────┼──────────────┘
          │                │                │                  │
          └────────────────┴────────────────┴──────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MVAS RUNTIME SERVER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Application Manager                              │   │
│  │  • Load/validate .mvapp packages                                     │   │
│  │  • Manage application lifecycle                                      │   │
│  │  • Hot-swap support                                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌──────────────────┬──────────────┴───────────────┬───────────────────┐   │
│  │                  │                              │                   │   │
│  ▼                  ▼                              ▼                   ▼   │
│ ┌────────────┐  ┌────────────┐  ┌────────────────────┐  ┌────────────┐    │
│ │   Input    │  │   Pre-     │  │    Inference       │  │   Post-    │    │
│ │  Manager   │  │ processing │  │     Engine         │  │ processing │    │
│ │            │  │  Pipeline  │  │                    │  │  & Output  │    │
│ │ • Cameras  │─▶│ • Resize   │─▶│ • ONNX Runtime     │─▶│ • Rules    │    │
│ │ • Folders  │  │ • Normalize│  │ • TorchScript      │  │ • API Resp │    │
│ │ • Streams  │  │ • Augment  │  │ • TensorRT         │  │ • PLC/GPIO │    │
│ │ • REST     │  │ • ROI Crop │  │ • OpenVINO         │  │ • MQTT     │    │
│ └────────────┘  └────────────┘  └────────────────────┘  └────────────┘    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Session & State Manager                          │   │
│  │  • Active sessions tracking                                          │   │
│  │  • Result caching                                                    │   │
│  │  • Statistics aggregation                                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            STORAGE LAYER                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Apps Storage   │  │  Results DB     │  │  Image Archive              │  │
│  │  (.mvapp files) │  │  (SQLite/PG)    │  │  (Local/S3/MinIO)           │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
Image Source ──▶ Input Manager ──▶ Preprocessing ──▶ Inference ──▶ Postprocessing ──▶ Output
     │                                                                                  │
     │                                                                                  ▼
     │                                                                          ┌──────────────┐
     │                                                                          │   Results    │
     └──────────────────────────────────────────────────────────────────────────│   Storage    │
                                              ▲                                 └──────────────┘
                                              │
                                         .mvapp file
                                     (defines all stages)
```

---

## 3. Architecture

### 3.1 Design Principles

1. **Separation of Concerns** - Each component has a single responsibility
2. **Plugin Architecture** - Easy to add new input sources, inference backends
3. **Configuration over Code** - Behavior defined in JSON/YAML, not code
4. **Async-First** - Non-blocking I/O for high throughput
5. **Stateless Processing** - Each request is independent (enables scaling)

### 3.2 Technology Stack

| Layer | Technology | Justification |
|-------|------------|---------------|
| **API Server** | FastAPI | Async, auto-docs, WebSocket support |
| **Inference** | ONNX Runtime | Cross-platform, optimized |
| **Camera SDK** | harvesters (GenICam) | Industry standard |
| **Queue** | asyncio.Queue | Simple, in-process |
| **Database** | SQLite / PostgreSQL | Embedded or scalable |
| **Caching** | LRU Cache | Fast model caching |

### 3.3 Module Dependency Graph

```
                    ┌─────────────────┐
                    │     server      │
                    │   (FastAPI)     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
      ┌───────────┐  ┌───────────────┐  ┌──────────┐
      │   REST    │  │   WebSocket   │  │  Static  │
      │  Routes   │  │   Handler     │  │  Files   │
      └─────┬─────┘  └───────┬───────┘  └──────────┘
            │                │
            └────────┬───────┘
                     ▼
            ┌────────────────┐
            │ AppManager     │
            └───────┬────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌──────────┐ ┌────────────┐ ┌────────────────┐
│AppLoader │ │SessionMgr  │ │ InferenceEngine│
└────┬─────┘ └────────────┘ └───────┬────────┘
     │                              │
     ▼                    ┌─────────┴─────────┐
┌──────────┐              │                   │
│ Manifest │              ▼                   ▼
│ Parser   │       ┌────────────┐     ┌─────────────┐
└──────────┘       │   ONNX     │     │ TorchScript │
                   │  Backend   │     │   Backend   │
                   └────────────┘     └─────────────┘
```

---

## 4. Component Design

### 4.1 Application Manager

**Responsibility:** Central coordinator for loading, managing, and running applications.

```python
class AppManager:
    """
    Singleton that manages all loaded applications.
    
    Responsibilities:
    - Load/unload .mvapp packages
    - Validate manifests
    - Manage inference engines per app
    - Track active sessions
    """
    
    def load_app(app_path: str) -> AppInstance
    def unload_app(app_id: str) -> bool
    def list_apps() -> List[AppInfo]
    def get_app(app_id: str) -> AppInstance
    def run_inference(app_id: str, image: np.ndarray) -> InferenceResult
```

### 4.2 Application Loader

**Responsibility:** Parse and validate `.mvapp` packages.

```python
class AppLoader:
    """
    Loads and validates .mvapp packages.
    
    Workflow:
    1. Extract ZIP to temp directory
    2. Parse manifest.json
    3. Validate all required files exist
    4. Load model into inference engine
    5. Return AppInstance
    """
    
    def load(app_path: str) -> AppInstance
    def validate_manifest(manifest: dict) -> ValidationResult
    def extract_package(app_path: str) -> str  # Returns temp dir
```

### 4.3 Inference Engine

**Responsibility:** Run model inference with multiple backend support.

```python
class InferenceEngine:
    """
    Abstract inference engine with pluggable backends.
    
    Supported Backends:
    - ONNX Runtime (default)
    - TorchScript (LibTorch)
    - TensorRT (NVIDIA)
    - OpenVINO (Intel)
    """
    
    def load_model(model_path: str, config: ModelConfig) -> None
    def infer(image: np.ndarray) -> InferenceOutput
    def warmup(iterations: int = 10) -> None
    def get_info() -> ModelInfo
```

### 4.4 Input Manager

**Responsibility:** Abstract interface for all image sources.

```python
class InputSource(ABC):
    """
    Abstract base class for all input sources.
    """
    
    @abstractmethod
    def connect() -> bool
    
    @abstractmethod
    def disconnect() -> None
    
    @abstractmethod
    def grab_image() -> np.ndarray
    
    @abstractmethod
    def start_streaming(callback: Callable) -> None
    
    @abstractmethod
    def stop_streaming() -> None
```

**Implementations:**

| Source | Class | Description |
|--------|-------|-------------|
| GigE Vision | `GigECamera` | Industrial cameras (GenICam) |
| USB Camera | `USBCamera` | Webcams, USB industrial cameras |
| RTSP Stream | `RTSPStream` | IP cameras, network streams |
| Folder Watch | `FolderWatcher` | Monitor folder for new images |
| REST Upload | `RESTUpload` | HTTP image upload endpoint |

### 4.5 Preprocessing Pipeline

**Responsibility:** Transform raw images before inference.

```python
class PreprocessingPipeline:
    """
    Configurable image preprocessing pipeline.
    
    Operations:
    - resize: Resize to target dimensions
    - crop: Crop to ROI
    - normalize: Apply mean/std normalization
    - to_tensor: Convert to model input format
    - augment: Apply augmentations (for training)
    """
    
    def __init__(self, transforms_config: dict)
    def process(image: np.ndarray) -> np.ndarray
```

### 4.6 Postprocessing & Output

**Responsibility:** Process inference results and route to outputs.

```python
class PostProcessor:
    """
    Apply decision rules and format output.
    """
    
    def apply_rules(inference_output: InferenceOutput) -> Decision
    def generate_visualization(image, output) -> np.ndarray
    def format_response(decision: Decision) -> dict

class OutputHandler(ABC):
    """
    Abstract output handler.
    """
    
    @abstractmethod
    def send(result: InspectionResult) -> None
```

---

## 5. Data Models

### 5.1 Core Models

```python
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class DecisionResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    REVIEW = "review"
    ERROR = "error"

class AppInfo(BaseModel):
    """Application metadata"""
    id: str
    name: str
    version: str
    description: str
    author: str
    created: datetime
    model_type: str
    input_resolution: tuple[int, int]

class ModelConfig(BaseModel):
    """Model configuration"""
    type: str  # "anomaly_detection", "classification", "segmentation"
    framework: str  # "onnx", "torchscript", "tensorrt"
    path: str
    device: str = "auto"  # "cpu", "cuda", "auto"
    precision: str = "fp32"  # "fp32", "fp16", "int8"

class InferenceOutput(BaseModel):
    """Raw inference output"""
    anomaly_score: float
    anomaly_map: Optional[np.ndarray] = None
    class_scores: Optional[Dict[str, float]] = None
    inference_time_ms: float

class InspectionResult(BaseModel):
    """Final inspection result"""
    request_id: str
    app_id: str
    timestamp: datetime
    decision: DecisionResult
    confidence: float
    details: Dict[str, Any]
    image_path: Optional[str] = None
    visualization_path: Optional[str] = None
    inference_time_ms: float
    total_time_ms: float

class CameraConfig(BaseModel):
    """Camera configuration"""
    type: str  # "gige", "usb", "rtsp", "folder"
    address: str  # IP, device ID, path, or URL
    settings: Dict[str, Any] = {}
```

### 5.2 API Request/Response Models

```python
class LoadAppRequest(BaseModel):
    app_path: str

class LoadAppResponse(BaseModel):
    success: bool
    app_id: str
    app_info: AppInfo
    message: str

class InspectRequest(BaseModel):
    app_id: str
    image_base64: Optional[str] = None
    image_url: Optional[str] = None

class InspectResponse(BaseModel):
    request_id: str
    decision: DecisionResult
    confidence: float
    anomaly_score: float
    inference_time_ms: float
    details: Dict[str, Any]
    visualization_base64: Optional[str] = None

class ConnectCameraRequest(BaseModel):
    camera_type: str
    address: str
    settings: Dict[str, Any] = {}

class StreamConfig(BaseModel):
    app_id: str
    camera_id: str
    fps: int = 30
    visualize: bool = True
```

---

## 6. API Specification

### 6.1 REST API Endpoints

#### Application Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/apps/load` | Load a .mvapp file |
| `DELETE` | `/api/v1/apps/{app_id}` | Unload an application |
| `GET` | `/api/v1/apps` | List all loaded apps |
| `GET` | `/api/v1/apps/{app_id}` | Get app details |
| `GET` | `/api/v1/apps/{app_id}/stats` | Get app statistics |

#### Inspection

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/inspect` | Run single image inspection |
| `POST` | `/api/v1/inspect/batch` | Run batch inspection |
| `GET` | `/api/v1/results/{request_id}` | Get inspection result |
| `GET` | `/api/v1/results` | Query inspection history |

#### Camera Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/cameras/connect` | Connect to camera |
| `DELETE` | `/api/v1/cameras/{camera_id}` | Disconnect camera |
| `GET` | `/api/v1/cameras` | List connected cameras |
| `GET` | `/api/v1/cameras/{camera_id}/grab` | Grab single frame |
| `POST` | `/api/v1/cameras/{camera_id}/settings` | Update camera settings |

#### Streaming

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/stream/start` | Start live inspection |
| `POST` | `/api/v1/stream/stop` | Stop live inspection |
| `GET` | `/api/v1/stream/status` | Get stream status |

### 6.2 WebSocket API

```
ws://server:8000/ws/stream/{session_id}

Messages (Server → Client):
- frame: { type: "frame", image_base64: "...", result: {...} }
- status: { type: "status", fps: 30, queue_size: 0 }
- error: { type: "error", message: "..." }

Messages (Client → Server):
- config: { type: "config", fps: 30, visualize: true }
- pause: { type: "pause" }
- resume: { type: "resume" }
```

### 6.3 Example API Usage

```bash
# Load an application
curl -X POST http://localhost:8000/api/v1/apps/load \
  -H "Content-Type: application/json" \
  -d '{"app_path": "/apps/bottle_inspection.mvapp"}'

# Response
{
  "success": true,
  "app_id": "bottle-inspection-v1",
  "app_info": {
    "name": "Bottle Cap Inspection",
    "version": "1.2.0",
    "model_type": "anomaly_detection"
  }
}

# Run inspection
curl -X POST http://localhost:8000/api/v1/inspect \
  -H "Content-Type: application/json" \
  -d '{
    "app_id": "bottle-inspection-v1",
    "image_base64": "/9j/4AAQSkZJRg..."
  }'

# Response
{
  "request_id": "req_abc123",
  "decision": "fail",
  "confidence": 0.95,
  "anomaly_score": 0.87,
  "inference_time_ms": 45.2,
  "details": {
    "defect_area_ratio": 0.023,
    "defect_location": [120, 85]
  }
}
```

---

## 7. Application Package Format

### 7.1 Package Structure

```
application.mvapp (ZIP file)
│
├── manifest.json              # Required: App metadata & config
│
├── model/                     # Required: Model files
│   ├── model.onnx            # Primary model file
│   ├── model_config.json     # Model-specific settings
│   └── memory_bank.pt        # Additional files (e.g., PatchCore memory)
│
├── preprocessing/             # Required: Preprocessing config
│   └── transforms.json       # Transform pipeline definition
│
├── postprocessing/            # Required: Decision rules
│   └── rules.json            # Threshold and decision rules
│
├── assets/                    # Optional: Reference images, masks
│   ├── reference.png         # Golden reference image
│   ├── roi_mask.png          # Region of interest mask
│   └── classes.json          # Class labels (for classification)
│
└── ui/                        # Optional: UI customization
    ├── overlay.json          # Visualization settings
    └── icon.png              # App icon for dashboard
```

### 7.2 Manifest Schema

```json
{
  "$schema": "https://mvas.io/schema/manifest-v1.json",
  "mvas_version": "1.0.0",
  
  "app": {
    "id": "unique-app-identifier",
    "name": "Human Readable Name",
    "version": "1.0.0",
    "description": "Detailed description of what this app does",
    "author": "Author Name",
    "created": "2025-12-12T00:00:00Z",
    "tags": ["anomaly", "manufacturing", "quality"]
  },
  
  "model": {
    "type": "anomaly_detection",
    "algorithm": "patchcore",
    "framework": "onnx",
    "path": "model/model.onnx",
    "additional_files": {
      "memory_bank": "model/memory_bank.pt"
    },
    "runtime": {
      "device": "auto",
      "precision": "fp16",
      "batch_size": 1,
      "num_threads": 4
    }
  },
  
  "input": {
    "type": "image",
    "color_mode": "RGB",
    "resolution": {
      "width": 256,
      "height": 256
    },
    "preprocessing": "preprocessing/transforms.json",
    "roi_mask": "assets/roi_mask.png"
  },
  
  "output": {
    "type": "anomaly_map",
    "postprocessing": "postprocessing/rules.json",
    "save_images": true,
    "save_visualizations": true
  },
  
  "requirements": {
    "min_mvas_version": "1.0.0",
    "gpu_required": false,
    "min_memory_mb": 512
  }
}
```

### 7.3 Transforms Schema

```json
{
  "pipeline": [
    {
      "op": "resize",
      "params": {
        "width": 256,
        "height": 256,
        "interpolation": "bilinear",
        "keep_aspect": false
      }
    },
    {
      "op": "crop_center",
      "params": {
        "width": 224,
        "height": 224
      },
      "enabled": false
    },
    {
      "op": "normalize",
      "params": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "scale": 255.0
      }
    },
    {
      "op": "to_tensor",
      "params": {
        "dtype": "float32",
        "channel_order": "CHW"
      }
    }
  ]
}
```

### 7.4 Rules Schema

```json
{
  "thresholds": {
    "anomaly_score": {
      "pass": 0.3,
      "review": 0.7
    }
  },
  
  "decision_logic": {
    "primary": "anomaly_score",
    "rules": [
      {
        "condition": "anomaly_score < thresholds.anomaly_score.pass",
        "decision": "pass",
        "confidence_source": "1 - anomaly_score"
      },
      {
        "condition": "anomaly_score >= thresholds.anomaly_score.pass AND anomaly_score < thresholds.anomaly_score.review",
        "decision": "review",
        "confidence_source": "anomaly_score"
      },
      {
        "condition": "anomaly_score >= thresholds.anomaly_score.review",
        "decision": "fail",
        "confidence_source": "anomaly_score"
      }
    ]
  },
  
  "outputs": {
    "include_anomaly_map": true,
    "include_bounding_boxes": true,
    "include_heatmap_overlay": true,
    "defect_min_area": 100
  },
  
  "visualization": {
    "heatmap_colormap": "jet",
    "heatmap_alpha": 0.5,
    "bbox_color": "#FF0000",
    "bbox_thickness": 2
  }
}
```

---

## 8. Input Sources

### 8.1 GigE Vision Camera

```python
class GigECamera(InputSource):
    """
    GigE Vision camera using GenICam/Harvester.
    
    Supported Features:
    - Auto-discovery
    - Hardware triggering
    - Exposure/gain control
    - Multiple pixel formats
    """
    
    def __init__(self, ip_address: str, config: dict):
        self.ip = ip_address
        self.config = config
        
    def connect(self) -> bool:
        # Use harvester to connect
        
    def configure(self, settings: dict):
        # Set exposure, gain, ROI, etc.
        
    def set_trigger_mode(self, mode: str):
        # "continuous", "software", "hardware"
```

**Configuration Example:**

```json
{
  "type": "gige",
  "address": "192.168.1.100",
  "settings": {
    "exposure_us": 10000,
    "gain_db": 0,
    "trigger_mode": "software",
    "pixel_format": "RGB8"
  }
}
```

### 8.2 USB Camera

```python
class USBCamera(InputSource):
    """
    USB camera using OpenCV VideoCapture.
    
    Supports:
    - Webcams
    - USB industrial cameras
    - DirectShow (Windows)
    - V4L2 (Linux)
    """
    
    def __init__(self, device_id: int, config: dict):
        self.device_id = device_id
        self.cap = cv2.VideoCapture(device_id)
```

### 8.3 Folder Watcher

```python
class FolderWatcher(InputSource):
    """
    Monitors a folder for new images.
    
    Use cases:
    - Offline batch processing
    - Integration with external systems
    - Testing and development
    """
    
    def __init__(self, folder_path: str, config: dict):
        self.folder = folder_path
        self.patterns = config.get("patterns", ["*.jpg", "*.png"])
        self.processed = set()
```

### 8.4 REST Upload

```python
class RESTUpload(InputSource):
    """
    Receives images via REST API upload.
    
    Accepts:
    - Base64 encoded images
    - Multipart file uploads
    - URL references
    """
```

---

## 9. Inference Pipeline

### 9.1 Pipeline Flow

```
┌──────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────┐
│  Input   │────▶│ Preprocessing│────▶│  Inference   │────▶│ Postproc  │
│  Image   │     │  Pipeline    │     │   Engine     │     │  Rules    │
└──────────┘     └──────────────┘     └──────────────┘     └───────────┘
                                              │
                                              ▼
                                    ┌──────────────────┐
                                    │  Backend Select  │
                                    │                  │
                                    │ ┌──────────────┐ │
                                    │ │ ONNX Runtime │ │
                                    │ ├──────────────┤ │
                                    │ │ TorchScript  │ │
                                    │ ├──────────────┤ │
                                    │ │ TensorRT     │ │
                                    │ ├──────────────┤ │
                                    │ │ OpenVINO     │ │
                                    │ └──────────────┘ │
                                    └──────────────────┘
```

### 9.2 ONNX Runtime Backend

```python
class ONNXBackend:
    def __init__(self, model_path: str, config: dict):
        providers = self._get_providers(config.get("device", "auto"))
        self.session = ort.InferenceSession(model_path, providers=providers)
        
    def _get_providers(self, device: str) -> list:
        if device == "auto":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cuda":
            return ["CUDAExecutionProvider"]
        else:
            return ["CPUExecutionProvider"]
            
    def infer(self, input_tensor: np.ndarray) -> dict:
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: input_tensor})
        return {"output": outputs[0]}
```

### 9.3 TorchScript Backend

```python
class TorchScriptBackend:
    def __init__(self, model_path: str, config: dict):
        self.device = self._get_device(config.get("device", "auto"))
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        
    def infer(self, input_tensor: np.ndarray) -> dict:
        with torch.no_grad():
            tensor = torch.from_numpy(input_tensor).to(self.device)
            output = self.model(tensor)
        return {"output": output.cpu().numpy()}
```

---

## 10. Output Handlers

### 10.1 Available Handlers

| Handler | Use Case | Protocol |
|---------|----------|----------|
| `RESTResponse` | API clients | HTTP/JSON |
| `WebSocketHandler` | Real-time UI | WebSocket |
| `PLCOutput` | Industrial automation | Modbus/OPC-UA |
| `GPIOHandler` | Hardware triggers | GPIO pins |
| `MQTTPublisher` | IoT integration | MQTT |
| `DatabaseLogger` | Result storage | SQL |
| `FileWriter` | Image archival | Filesystem |

### 10.2 PLC Integration

```python
class PLCOutput(OutputHandler):
    """
    Send results to PLC via Modbus TCP.
    
    Registers:
    - 0: Result code (0=pass, 1=fail, 2=review)
    - 1: Anomaly score (0-1000, scaled)
    - 2: Inspection count
    """
    
    def __init__(self, plc_ip: str, port: int = 502):
        self.client = ModbusTcpClient(plc_ip, port)
        
    def send(self, result: InspectionResult):
        self.client.write_register(0, result.decision.value)
        self.client.write_register(1, int(result.confidence * 1000))
```

---

## 11. Deployment

### 11.1 Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY mvas_runtime/ ./mvas_runtime/
COPY config/ ./config/

EXPOSE 8000
CMD ["uvicorn", "mvas_runtime.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 11.2 Docker Compose (Full Stack)

```yaml
version: "3.8"

services:
  mvas-server:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./apps:/apps
      - ./data:/data
    environment:
      - MVAS_APPS_DIR=/apps
      - MVAS_DATA_DIR=/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  mvas-ui:
    image: mvas/dashboard:latest
    ports:
      - "3000:3000"
    depends_on:
      - mvas-server

  database:
    image: postgres:15
    environment:
      POSTGRES_DB: mvas
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

### 11.3 Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mvas-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mvas-server
  template:
    spec:
      containers:
      - name: mvas
        image: mvas/server:1.0.0
        resources:
          limits:
            nvidia.com/gpu: 1
        ports:
        - containerPort: 8000
```

---

## 12. Security

### 12.1 Authentication

- API Key authentication for REST endpoints
- JWT tokens for session management
- Role-based access control (Operator, Engineer, Admin)

### 12.2 Network Security

- TLS/HTTPS for all communications
- Camera network isolation (separate VLAN)
- Firewall rules for PLC communication

### 12.3 Application Security

- Package signing for `.mvapp` files
- Sandboxed execution environment
- Input validation on all endpoints

---

## 13. Performance

### 13.1 Benchmarks

| Metric | Target | Notes |
|--------|--------|-------|
| **Single inference** | < 50ms | GPU, FP16 |
| **End-to-end latency** | < 100ms | Camera to result |
| **Throughput** | > 30 FPS | Per camera |
| **Memory per app** | < 500MB | Excluding GPU |
| **App load time** | < 5s | Cold start |

### 13.2 Optimization Strategies

1. **Model Optimization**
   - TensorRT conversion for NVIDIA GPUs
   - OpenVINO for Intel CPUs
   - FP16/INT8 quantization

2. **Batching**
   - Accumulate frames for batch inference
   - Dynamic batch sizing

3. **Caching**
   - LRU cache for loaded models
   - Result caching for identical images

4. **Async Processing**
   - Non-blocking image acquisition
   - Pipeline parallelization

---

## 14. Future Roadmap

### Phase 1 (Current) - Basic Runtime
- [x] Core runtime implementation
- [x] ONNX/TorchScript support
- [x] Basic camera support (USB, folder)
- [x] REST API

### Phase 2 (Q1 2026) - Production Ready
- [ ] GigE Vision camera support
- [ ] Web dashboard UI
- [ ] Docker deployment
- [ ] Result database

### Phase 3 (Q2 2026) - Enhanced Pipeline (v2.0)
- [ ] **DAG-based pipeline execution** (see [ENHANCED_ARCHITECTURE.md](./ENHANCED_ARCHITECTURE.md))
- [ ] **Multi-model support** with branching and loops
- [ ] **Sandboxed custom processors** (Python)
- [ ] TensorRT optimization
- [ ] PLC/Modbus integration
- [ ] Multi-camera support

### Phase 4 (Q3 2026) - App Marketplace
- [ ] **MVAS App Store** - browse, download, install apps
- [ ] **Developer Portal** - publish and monetize apps
- [ ] **Package signing** and security verification
- [ ] Kubernetes deployment

### Phase 5 (Q4 2026) - Enterprise & Edge
- [ ] Active learning feedback loop
- [ ] Model retraining pipeline
- [ ] Edge deployment (NVIDIA Jetson)
- [ ] Multi-tenant cloud version
- [ ] Enterprise license management

---

## Appendix A: Error Codes

| Code | Name | Description |
|------|------|-------------|
| `E1001` | APP_NOT_FOUND | Application file not found |
| `E1002` | APP_INVALID | Invalid manifest or corrupted package |
| `E1003` | APP_LOAD_FAILED | Failed to load model |
| `E2001` | CAMERA_NOT_FOUND | Camera not found at address |
| `E2002` | CAMERA_CONNECT_FAILED | Failed to connect to camera |
| `E2003` | CAMERA_GRAB_FAILED | Failed to grab frame |
| `E3001` | INFERENCE_FAILED | Model inference error |
| `E3002` | PREPROCESSING_FAILED | Image preprocessing error |
| `E4001` | SESSION_NOT_FOUND | Invalid session ID |
| `E4002` | SESSION_EXPIRED | Session timeout |

---

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **MVAS** | Machine Vision Application Standard |
| **mvapp** | MVAS application package file format |
| **GenICam** | Generic Interface for Cameras standard |
| **GigE Vision** | Camera interface standard over Ethernet |
| **ROI** | Region of Interest |
| **Anomaly Map** | Heatmap showing detected anomalies |

---

*End of System Design Document*

