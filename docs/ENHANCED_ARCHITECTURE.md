# MVAS Enhanced Architecture
## Multi-Stage Pipeline & App Marketplace Design

**Version:** 2.0.0  
**Date:** December 16, 2025  
**Status:** Proposed Enhancement

---

## 1. Problem Statement

The current MVAS design has critical limitations for real-world industrial machine vision:

### Current Limitations

```
Current Simple Pipeline:
Image → Preprocess → Single Model → Postprocess → Output
```

### Real-World Requirements

```
Complex Multi-Stage Pipeline Example:
                                    ┌─────────────────┐
                                    │  Classification │──→ Class A Processing
                                    │     Model       │──→ Class B Processing
                                    └────────┬────────┘
                                             │
Image → Preprocess → Segmentation → ROI Extract → Anomaly Detection → Post → Output
                          │              │                │
                          │              ├──→ Region 1 ───┤
                          │              ├──→ Region 2 ───┤
                          │              └──→ Region N ───┘
                          │
                          └──→ Quality Metrics → Statistical Analysis
```

**Real Use Cases:**
1. **PCB Inspection**: Locate components → Check each component → Solder joint analysis → OCR reading
2. **Bottle Inspection**: Cap detection → Cap quality → Fill level → Label OCR → Final decision
3. **Automotive Parts**: Surface segmentation → Defect detection per region → Measurement → Classification

---

## 2. Enhanced Architecture Overview

### 2.1 DAG-Based Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MVAS ENHANCED RUNTIME                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                      Pipeline Execution Engine (DAG)                       │ │
│  │                                                                           │ │
│  │   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐              │ │
│  │   │  Node   │───▶│  Node   │───▶│  Node   │───▶│  Node   │              │ │
│  │   │ (Input) │    │ (Model1)│    │ (Model2)│    │(Output) │              │ │
│  │   └─────────┘    └────┬────┘    └─────────┘    └─────────┘              │ │
│  │                       │              ▲                                    │ │
│  │                       │              │                                    │ │
│  │                       └──────────────┴── Branch/Merge Logic               │ │
│  │                                                                           │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐   │
│  │ Node Types    │  │ Data Context  │  │ Condition     │  │ Sandbox       │   │
│  │               │  │               │  │ Evaluator     │  │ Runtime       │   │
│  │ • InputNode   │  │ • Images      │  │               │  │               │   │
│  │ • ModelNode   │  │ • Tensors     │  │ • If/Else     │  │ • WASM        │   │
│  │ • ProcessNode │  │ • Metadata    │  │ • Switch      │  │ • Python      │   │
│  │ • BranchNode  │  │ • Results     │  │ • Loop        │  │ • Container   │   │
│  │ • MergeNode   │  │ • State       │  │ • ForEach     │  │               │   │
│  │ • OutputNode  │  │               │  │               │  │               │   │
│  └───────────────┘  └───────────────┘  └───────────────┘  └───────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Node Type Definitions

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class NodeType(Enum):
    INPUT = "input"           # Image/data source
    PREPROCESS = "preprocess" # Image transforms
    MODEL = "model"           # ML model inference
    PROCESS = "process"       # Custom processing logic
    BRANCH = "branch"         # Conditional routing
    MERGE = "merge"           # Combine multiple inputs
    LOOP = "loop"             # Iterate over regions/items
    OUTPUT = "output"         # Final output/decision

@dataclass
class PipelineContext:
    """Shared context passed through pipeline"""
    original_image: np.ndarray
    current_data: Dict[str, Any]      # Current processing data
    metadata: Dict[str, Any]          # Image metadata, timestamps
    results: Dict[str, Any]           # Accumulated results from nodes
    state: Dict[str, Any]             # Mutable state for complex logic
    
class PipelineNode(ABC):
    """Base class for all pipeline nodes"""
    
    def __init__(self, node_id: str, config: dict):
        self.node_id = node_id
        self.config = config
        self.inputs: List[str] = []   # Input node IDs
        self.outputs: List[str] = []  # Output node IDs
    
    @abstractmethod
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """Execute node logic and return updated context"""
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """Validate node configuration"""
        pass

class ModelNode(PipelineNode):
    """Node that runs ML model inference"""
    
    def __init__(self, node_id: str, config: dict):
        super().__init__(node_id, config)
        self.model_path = config["model_path"]
        self.input_key = config.get("input_key", "image")
        self.output_key = config.get("output_key", "prediction")
        self.engine = None
        
    async def execute(self, context: PipelineContext) -> PipelineContext:
        input_data = context.current_data[self.input_key]
        output = await self.engine.infer(input_data)
        context.results[self.output_key] = output
        return context

class BranchNode(PipelineNode):
    """Conditional branching based on results"""
    
    def __init__(self, node_id: str, config: dict):
        super().__init__(node_id, config)
        self.conditions = config["conditions"]  # List of condition rules
        
    async def execute(self, context: PipelineContext) -> PipelineContext:
        for condition in self.conditions:
            if self._evaluate_condition(condition["expr"], context):
                context.state["next_branch"] = condition["target"]
                break
        return context
    
    def _evaluate_condition(self, expr: str, context: PipelineContext) -> bool:
        # Safe expression evaluation
        return eval_safe(expr, context.results)

class LoopNode(PipelineNode):
    """Iterate over detected regions/items"""
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        items = context.results.get(self.config["items_key"], [])
        sub_results = []
        
        for item in items:
            sub_context = context.copy()
            sub_context.current_data["current_item"] = item
            # Execute sub-pipeline for each item
            result = await self._execute_sub_pipeline(sub_context)
            sub_results.append(result)
            
        context.results[self.config["output_key"]] = sub_results
        return context
```

---

## 3. Enhanced Pipeline Definition Schema

### 3.1 Pipeline Manifest (pipeline.json)

```json
{
  "pipeline_version": "2.0",
  "name": "PCB Multi-Stage Inspection",
  "description": "Complete PCB inspection with component detection, solder analysis, and OCR",
  
  "nodes": {
    "input": {
      "type": "input",
      "config": {
        "sources": ["camera", "folder", "api"]
      }
    },
    
    "preprocess": {
      "type": "preprocess",
      "depends_on": ["input"],
      "config": {
        "transforms": "preprocessing/transforms.json"
      }
    },
    
    "component_detector": {
      "type": "model",
      "depends_on": ["preprocess"],
      "config": {
        "model_path": "models/component_detector.onnx",
        "model_type": "object_detection",
        "input_key": "preprocessed_image",
        "output_key": "detected_components"
      }
    },
    
    "component_loop": {
      "type": "loop",
      "depends_on": ["component_detector"],
      "config": {
        "items_key": "detected_components.boxes",
        "output_key": "component_results",
        "sub_pipeline": {
          "crop_roi": {
            "type": "process",
            "config": {
              "operation": "crop_to_bbox",
              "padding": 10
            }
          },
          "classify_component": {
            "type": "model",
            "depends_on": ["crop_roi"],
            "config": {
              "model_path": "models/component_classifier.onnx",
              "output_key": "component_class"
            }
          },
          "branch_by_type": {
            "type": "branch",
            "depends_on": ["classify_component"],
            "config": {
              "conditions": [
                {
                  "expr": "component_class == 'resistor'",
                  "target": "resistor_check"
                },
                {
                  "expr": "component_class == 'capacitor'",
                  "target": "capacitor_check"
                },
                {
                  "expr": "component_class == 'ic_chip'",
                  "target": "ic_check"
                }
              ]
            }
          },
          "resistor_check": {
            "type": "model",
            "config": {
              "model_path": "models/resistor_anomaly.onnx"
            }
          },
          "capacitor_check": {
            "type": "model",
            "config": {
              "model_path": "models/capacitor_anomaly.onnx"
            }
          },
          "ic_check": {
            "type": "pipeline",
            "config": {
              "sub_nodes": {
                "solder_check": {
                  "type": "model",
                  "config": {"model_path": "models/solder_detector.onnx"}
                },
                "pin_alignment": {
                  "type": "model",
                  "depends_on": ["solder_check"],
                  "config": {"model_path": "models/pin_alignment.onnx"}
                },
                "ocr_read": {
                  "type": "model",
                  "depends_on": ["solder_check"],
                  "config": {"model_path": "models/ic_ocr.onnx"}
                }
              }
            }
          }
        }
      }
    },
    
    "global_quality": {
      "type": "model",
      "depends_on": ["preprocess"],
      "config": {
        "model_path": "models/global_quality.onnx",
        "output_key": "overall_quality"
      }
    },
    
    "merge_results": {
      "type": "merge",
      "depends_on": ["component_loop", "global_quality"],
      "config": {
        "strategy": "aggregate",
        "inputs": ["component_results", "overall_quality"]
      }
    },
    
    "decision_engine": {
      "type": "process",
      "depends_on": ["merge_results"],
      "config": {
        "processor": "processors/decision_rules.py",
        "rules": "rules/final_decision.json"
      }
    },
    
    "output": {
      "type": "output",
      "depends_on": ["decision_engine"],
      "config": {
        "handlers": ["api_response", "plc_signal", "database"]
      }
    }
  },
  
  "execution": {
    "mode": "dag",
    "parallel_branches": true,
    "timeout_seconds": 30,
    "error_handling": "continue_on_warning"
  }
}
```

### 3.2 Visual Pipeline Representation

```
                                    ┌──────────────────┐
                                    │  Global Quality  │
                                    │     Model        │
                                    └────────┬─────────┘
                                             │
┌───────┐   ┌────────────┐   ┌────────────┐  │  ┌─────────────┐   ┌──────────┐   ┌────────┐
│ Input │──▶│ Preprocess │──▶│ Component  │──┼─▶│   Merge     │──▶│ Decision │──▶│ Output │
│       │   │            │   │  Detector  │  │  │   Results   │   │  Engine  │   │        │
└───────┘   └────────────┘   └─────┬──────┘  │  └─────────────┘   └──────────┘   └────────┘
                                   │         │         ▲
                                   ▼         │         │
                           ┌───────────────┐ │         │
                           │  Loop: Each   │ │         │
                           │  Component    │─┘         │
                           └───────┬───────┘           │
                                   │                   │
                    ┌──────────────┼──────────────┐    │
                    │              │              │    │
                    ▼              ▼              ▼    │
             ┌───────────┐  ┌───────────┐  ┌─────────┐│
             │ Resistor  │  │ Capacitor │  │   IC    ││
             │   Check   │  │   Check   │  │  Check  ││
             └─────┬─────┘  └─────┬─────┘  └────┬────┘│
                   │              │              │     │
                   └──────────────┴──────────────┴─────┘
```

---

## 4. Executable Application Package (.mvapp v2)

### 4.1 Enhanced Package Structure

The `.mvapp` format becomes a true executable package, similar to APK or Chrome extensions:

```
bottle_inspection.mvapp (Signed ZIP)
│
├── manifest.json              # App metadata, permissions, requirements
├── signature.json             # Digital signature for verification
│
├── pipeline/                  # Pipeline definition
│   ├── pipeline.json          # DAG pipeline definition
│   └── sub_pipelines/         # Reusable sub-pipelines
│       └── solder_check.json
│
├── models/                    # All ML models
│   ├── segmentation.onnx
│   ├── anomaly_detector.onnx
│   ├── classifier.onnx
│   └── configs/
│       └── model_configs.json
│
├── processors/                # Custom processing scripts (sandboxed)
│   ├── decision_rules.py      # Python decision logic
│   ├── measurement.py         # Custom measurements
│   └── validators/
│       └── quality_checks.py
│
├── rules/                     # Decision and business rules
│   ├── thresholds.json
│   ├── decision_tree.json
│   └── plc_mapping.json
│
├── assets/                    # Static assets
│   ├── reference_images/
│   ├── calibration/
│   └── ui/
│       ├── icon.png
│       ├── preview.png
│       └── dashboard.json
│
├── localization/              # Multi-language support
│   ├── en.json
│   ├── zh.json
│   └── de.json
│
├── tests/                     # Built-in test cases
│   ├── test_images/
│   ├── expected_results.json
│   └── benchmark_config.json
│
└── runtime/                   # Optional: Custom runtime components
    ├── requirements.txt       # Python dependencies
    └── lib/                   # Compiled libraries (WASM/native)
```

### 4.2 Enhanced Manifest Schema

```json
{
  "$schema": "https://mvas.io/schema/manifest-v2.json",
  "manifest_version": "2.0.0",
  
  "app": {
    "id": "com.company.bottle-inspection",
    "name": "Advanced Bottle Inspection Suite",
    "version": "2.1.0",
    "version_code": 21,
    "description": "Complete bottle inspection including cap, fill level, and label",
    "author": {
      "name": "Vision Systems Inc.",
      "email": "support@visionsystems.com",
      "url": "https://visionsystems.com"
    },
    "license": "commercial",
    "created": "2025-12-16T00:00:00Z",
    "updated": "2025-12-16T00:00:00Z",
    "tags": ["bottle", "beverage", "manufacturing", "anomaly"],
    "category": "quality_inspection",
    "icon": "assets/ui/icon.png",
    "preview_images": [
      "assets/ui/preview.png",
      "assets/ui/result_example.png"
    ]
  },
  
  "pipeline": {
    "definition": "pipeline/pipeline.json",
    "type": "dag",
    "models_count": 4,
    "stages": ["segmentation", "classification", "anomaly_detection", "measurement"]
  },
  
  "models": [
    {
      "id": "cap_segmentation",
      "path": "models/cap_seg.onnx",
      "type": "segmentation",
      "framework": "onnx",
      "input_shape": [1, 3, 512, 512],
      "output_type": "mask"
    },
    {
      "id": "cap_anomaly",
      "path": "models/cap_anomaly.onnx",
      "type": "anomaly_detection",
      "framework": "onnx",
      "algorithm": "patchcore"
    },
    {
      "id": "fill_level",
      "path": "models/fill_level.onnx",
      "type": "regression",
      "framework": "onnx"
    },
    {
      "id": "label_ocr",
      "path": "models/label_ocr.onnx",
      "type": "ocr",
      "framework": "onnx"
    }
  ],
  
  "processors": [
    {
      "id": "decision_engine",
      "path": "processors/decision_rules.py",
      "entry_point": "make_decision",
      "sandbox": "python_restricted",
      "permissions": ["read_results", "write_output"]
    }
  ],
  
  "permissions": {
    "required": [
      "camera.read",
      "results.write",
      "network.local"
    ],
    "optional": [
      "plc.write",
      "database.write",
      "network.external"
    ]
  },
  
  "requirements": {
    "mvas_version": ">=2.0.0",
    "runtime": {
      "min_memory_mb": 1024,
      "gpu_memory_mb": 512,
      "gpu_required": false,
      "gpu_recommended": true
    },
    "dependencies": {
      "onnxruntime": ">=1.15.0"
    }
  },
  
  "configuration": {
    "user_configurable": [
      {
        "key": "thresholds.anomaly_score",
        "type": "float",
        "default": 0.5,
        "min": 0.0,
        "max": 1.0,
        "label": "Anomaly Threshold",
        "description": "Score above this value marks as defective"
      },
      {
        "key": "thresholds.fill_level_min",
        "type": "float",
        "default": 0.95,
        "label": "Minimum Fill Level",
        "description": "Minimum acceptable fill level (0-1)"
      }
    ]
  },
  
  "testing": {
    "test_suite": "tests/",
    "benchmark_images": "tests/test_images/",
    "expected_results": "tests/expected_results.json",
    "performance_baseline": {
      "inference_time_ms": 150,
      "throughput_fps": 10
    }
  },
  
  "marketplace": {
    "price": "free",
    "trial_days": 30,
    "support_url": "https://visionsystems.com/support",
    "documentation_url": "https://docs.visionsystems.com/bottle",
    "video_tutorial": "https://youtube.com/watch?v=xxx"
  }
}
```

---

## 5. MVAS App Marketplace Architecture

### 5.1 Marketplace Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           MVAS MARKETPLACE (Cloud)                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │   App Store     │    │  Developer      │    │     Review & Security       │ │
│  │   Frontend      │    │  Portal         │    │     Pipeline                │ │
│  │                 │    │                 │    │                             │ │
│  │ • Browse apps   │    │ • Upload apps   │    │ • Automated scanning        │ │
│  │ • Search/filter │    │ • Manage apps   │    │ • Model validation          │ │
│  │ • Reviews       │    │ • Analytics     │    │ • Security audit            │ │
│  │ • Downloads     │    │ • Versioning    │    │ • Performance benchmarks    │ │
│  │ • Ratings       │    │ • Monetization  │    │ • Human review (optional)   │ │
│  └────────┬────────┘    └────────┬────────┘    └──────────────┬──────────────┘ │
│           │                      │                            │                 │
│           └──────────────────────┴────────────────────────────┘                 │
│                                  │                                              │
│                                  ▼                                              │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                        App Repository                                      │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐   │ │
│  │  │ App Storage │  │ Metadata    │  │ User Data   │  │ Analytics       │   │ │
│  │  │ (S3/Blob)   │  │ (Database)  │  │ (Reviews)   │  │ (Usage Stats)   │   │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │  HTTPS/API
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        MVAS LOCAL RUNTIME (Factory Floor)                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐ │
│  │   App Manager   │    │  Update         │    │     License Manager         │ │
│  │                 │    │  Manager        │    │                             │ │
│  │ • Install apps  │    │ • Check updates │    │ • Validate licenses         │ │
│  │ • Run apps      │    │ • Auto-update   │    │ • Offline activation        │ │
│  │ • Configure     │    │ • Rollback      │    │ • Usage tracking            │ │
│  └─────────────────┘    └─────────────────┘    └─────────────────────────────┘ │
│                                                                                 │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                     Pipeline Execution Engine                              │ │
│  │                      (Runs installed .mvapp)                               │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Developer Publishing Flow

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Developer Publishing Workflow                          │
└──────────────────────────────────────────────────────────────────────────────┘

  Developer                    Marketplace                    End User
      │                            │                              │
      │  1. Create .mvapp          │                              │
      │────────────────────▶       │                              │
      │                            │                              │
      │  2. Upload to Portal       │                              │
      │────────────────────▶       │                              │
      │                            │                              │
      │                     3. Automated Review                   │
      │                     ┌──────────────┐                      │
      │                     │ • Virus scan │                      │
      │                     │ • Model test │                      │
      │                     │ • Benchmark  │                      │
      │                     │ • Signature  │                      │
      │                     └──────────────┘                      │
      │                            │                              │
      │  4. Review Results         │                              │
      │◀────────────────────       │                              │
      │                            │                              │
      │  5. Publish/Fix            │                              │
      │────────────────────▶       │                              │
      │                            │                              │
      │                     6. Live in Store                      │
      │                            │                              │
      │                            │  7. Browse & Download        │
      │                            │◀─────────────────────────────│
      │                            │                              │
      │                            │  8. Install & Run            │
      │                            │─────────────────────────────▶│
      │                            │                              │
      │  9. Usage Analytics        │                              │
      │◀────────────────────       │                              │
      │                            │                              │
      │                            │  10. Review/Rating           │
      │                            │◀─────────────────────────────│
```

### 5.3 Marketplace API

```python
# Marketplace REST API

# Developer APIs
POST   /api/v1/developer/register          # Register as developer
POST   /api/v1/developer/apps              # Upload new app
PUT    /api/v1/developer/apps/{app_id}     # Update app
GET    /api/v1/developer/apps/{app_id}/stats  # Get analytics
POST   /api/v1/developer/apps/{app_id}/release # Release new version

# Store APIs
GET    /api/v1/store/apps                  # List all apps
GET    /api/v1/store/apps/{app_id}         # Get app details
GET    /api/v1/store/apps/{app_id}/download # Download app
GET    /api/v1/store/categories            # List categories
GET    /api/v1/store/search?q={query}      # Search apps
GET    /api/v1/store/featured              # Featured apps

# User APIs
POST   /api/v1/user/apps/{app_id}/review   # Submit review
GET    /api/v1/user/purchased              # List purchased apps
POST   /api/v1/user/apps/{app_id}/report   # Report issue

# Local Runtime APIs
POST   /api/v1/runtime/install             # Install app from store
DELETE /api/v1/runtime/apps/{app_id}       # Uninstall app
GET    /api/v1/runtime/apps                # List installed apps
POST   /api/v1/runtime/apps/{app_id}/run   # Run app
GET    /api/v1/runtime/updates             # Check for updates
```

### 5.4 App Store Data Model

```python
from pydantic import BaseModel
from datetime import datetime
from typing import List, Optional
from enum import Enum

class AppCategory(str, Enum):
    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    OBJECT_DETECTION = "object_detection"
    SEGMENTATION = "segmentation"
    OCR = "ocr"
    MEASUREMENT = "measurement"
    QUALITY_INSPECTION = "quality_inspection"

class AppStatus(str, Enum):
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"
    DEPRECATED = "deprecated"

class PricingModel(str, Enum):
    FREE = "free"
    ONE_TIME = "one_time"
    SUBSCRIPTION = "subscription"
    PER_INFERENCE = "per_inference"

class MarketplaceApp(BaseModel):
    """App listing in marketplace"""
    id: str
    name: str
    description: str
    version: str
    version_code: int
    author_id: str
    author_name: str
    category: AppCategory
    tags: List[str]
    icon_url: str
    preview_images: List[str]
    
    # Metrics
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0
    
    # Pricing
    pricing: PricingModel = PricingModel.FREE
    price_usd: Optional[float] = None
    
    # Status
    status: AppStatus = AppStatus.DRAFT
    published_at: Optional[datetime] = None
    updated_at: datetime
    
    # Technical
    package_size_mb: float
    min_mvas_version: str
    gpu_required: bool = False
    
    # Links
    documentation_url: Optional[str] = None
    support_url: Optional[str] = None

class AppReview(BaseModel):
    """User review for an app"""
    id: str
    app_id: str
    user_id: str
    rating: int  # 1-5
    title: str
    content: str
    created_at: datetime
    helpful_votes: int = 0
    
class AppVersion(BaseModel):
    """Version history for an app"""
    version: str
    version_code: int
    release_notes: str
    package_url: str
    created_at: datetime
    downloads: int = 0
```

---

## 6. Security & Sandboxing

### 6.1 Sandboxed Execution

Custom processors in `.mvapp` packages run in a sandboxed environment:

```python
class ProcessorSandbox:
    """
    Sandboxed execution environment for custom processors.
    
    Restrictions:
    - No filesystem access outside app directory
    - No network access (unless explicitly permitted)
    - No subprocess execution
    - Limited memory allocation
    - Timeout enforcement
    """
    
    RESTRICTED_IMPORTS = [
        'os', 'subprocess', 'socket', 'http', 'urllib',
        'shutil', 'pathlib', 'tempfile', '__builtins__'
    ]
    
    ALLOWED_IMPORTS = [
        'numpy', 'cv2', 'math', 'json', 'typing',
        'dataclasses', 'enum', 'collections'
    ]
    
    def execute(self, code: str, context: dict, timeout: float = 5.0):
        """Execute code in sandbox with restrictions"""
        restricted_globals = self._create_restricted_globals()
        
        with resource_limits(memory_mb=256, cpu_seconds=timeout):
            exec(code, restricted_globals, context)
        
        return context
```

### 6.2 Package Signing

```python
class PackageSigner:
    """
    Sign and verify .mvapp packages.
    
    Uses asymmetric encryption:
    - Developer signs with private key
    - Runtime verifies with public key from marketplace
    """
    
    def sign_package(self, package_path: str, private_key: bytes) -> str:
        """Sign package and return signature"""
        content_hash = self._compute_package_hash(package_path)
        signature = self._sign_hash(content_hash, private_key)
        return signature
    
    def verify_package(self, package_path: str, signature: str, public_key: bytes) -> bool:
        """Verify package signature"""
        content_hash = self._compute_package_hash(package_path)
        return self._verify_signature(content_hash, signature, public_key)
```

---

## 7. Pipeline Execution Engine

### 7.1 DAG Executor

```python
import asyncio
from typing import Dict, List, Set
from dataclasses import dataclass

@dataclass
class ExecutionPlan:
    """Optimized execution plan from DAG analysis"""
    stages: List[List[str]]  # Parallel stages
    dependencies: Dict[str, Set[str]]
    
class DAGExecutor:
    """
    Execute pipeline DAG with parallel stage support.
    """
    
    def __init__(self, pipeline_config: dict):
        self.nodes = self._build_nodes(pipeline_config)
        self.plan = self._create_execution_plan()
        
    def _create_execution_plan(self) -> ExecutionPlan:
        """Analyze DAG and create optimal execution plan"""
        # Topological sort with parallel stage identification
        stages = []
        remaining = set(self.nodes.keys())
        completed = set()
        
        while remaining:
            # Find all nodes with satisfied dependencies
            ready = {
                node_id for node_id in remaining
                if all(dep in completed for dep in self.nodes[node_id].inputs)
            }
            
            if not ready:
                raise ValueError("Circular dependency detected")
            
            stages.append(list(ready))
            completed.update(ready)
            remaining -= ready
            
        return ExecutionPlan(stages=stages, dependencies=self._get_dependencies())
    
    async def execute(self, initial_context: PipelineContext) -> PipelineContext:
        """Execute pipeline following the execution plan"""
        context = initial_context
        
        for stage in self.plan.stages:
            if len(stage) == 1:
                # Single node, execute directly
                context = await self.nodes[stage[0]].execute(context)
            else:
                # Multiple nodes can run in parallel
                tasks = [
                    self.nodes[node_id].execute(context.copy())
                    for node_id in stage
                ]
                results = await asyncio.gather(*tasks)
                context = self._merge_contexts(results)
        
        return context
    
    def _merge_contexts(self, contexts: List[PipelineContext]) -> PipelineContext:
        """Merge results from parallel execution"""
        merged = contexts[0].copy()
        for ctx in contexts[1:]:
            merged.results.update(ctx.results)
        return merged
```

### 7.2 Real-World Pipeline Example

```python
# Example: Automotive Part Inspection Pipeline

pipeline_config = {
    "nodes": {
        # Stage 1: Input
        "camera_input": {
            "type": "input",
            "config": {"source": "gige_camera", "ip": "192.168.1.100"}
        },
        
        # Stage 2: Preprocessing
        "preprocess": {
            "type": "preprocess",
            "depends_on": ["camera_input"],
            "config": {
                "resize": [1024, 1024],
                "normalize": True
            }
        },
        
        # Stage 3: Run in parallel
        "surface_segmentation": {
            "type": "model",
            "depends_on": ["preprocess"],
            "config": {"model": "models/surface_seg.onnx"}
        },
        "edge_detection": {
            "type": "model", 
            "depends_on": ["preprocess"],
            "config": {"model": "models/edge_detect.onnx"}
        },
        "color_analysis": {
            "type": "process",
            "depends_on": ["preprocess"],
            "config": {"processor": "processors/color_histogram.py"}
        },
        
        # Stage 4: Process segmented regions
        "region_loop": {
            "type": "loop",
            "depends_on": ["surface_segmentation"],
            "config": {
                "items_key": "regions",
                "sub_pipeline": {
                    "anomaly_check": {
                        "type": "model",
                        "config": {"model": "models/anomaly.onnx"}
                    },
                    "measurement": {
                        "type": "process",
                        "config": {"processor": "processors/measure.py"}
                    }
                }
            }
        },
        
        # Stage 5: Merge all results
        "merge": {
            "type": "merge",
            "depends_on": ["region_loop", "edge_detection", "color_analysis"],
            "config": {"strategy": "aggregate_all"}
        },
        
        # Stage 6: Final decision
        "decision": {
            "type": "process",
            "depends_on": ["merge"],
            "config": {
                "processor": "processors/decision_tree.py",
                "rules": "rules/pass_fail.json"
            }
        },
        
        # Stage 7: Output
        "output": {
            "type": "output",
            "depends_on": ["decision"],
            "config": {
                "handlers": ["plc", "database", "mqtt"]
            }
        }
    }
}

# Execution plan created:
# Stage 1: [camera_input]
# Stage 2: [preprocess]  
# Stage 3: [surface_segmentation, edge_detection, color_analysis]  <- Parallel!
# Stage 4: [region_loop]
# Stage 5: [merge]
# Stage 6: [decision]
# Stage 7: [output]
```

---

## 8. Migration Path

### 8.1 Backward Compatibility

Existing v1 `.mvapp` packages will still work:

```python
class AppLoader:
    def load(self, package_path: str) -> Application:
        manifest = self._read_manifest(package_path)
        
        if manifest.get("manifest_version", "1.0") < "2.0":
            # Convert v1 to v2 format
            return self._load_v1_app(package_path, manifest)
        else:
            return self._load_v2_app(package_path, manifest)
    
    def _load_v1_app(self, path: str, manifest: dict) -> Application:
        """Convert v1 simple pipeline to v2 DAG format"""
        v2_pipeline = {
            "nodes": {
                "input": {"type": "input"},
                "preprocess": {
                    "type": "preprocess",
                    "depends_on": ["input"],
                    "config": manifest.get("input", {})
                },
                "model": {
                    "type": "model",
                    "depends_on": ["preprocess"],
                    "config": manifest.get("model", {})
                },
                "postprocess": {
                    "type": "process",
                    "depends_on": ["model"],
                    "config": manifest.get("output", {})
                },
                "output": {
                    "type": "output",
                    "depends_on": ["postprocess"]
                }
            }
        }
        return self._build_application(v2_pipeline)
```

---

## 9. Summary

### Key Enhancements

| Feature | v1 (Current) | v2 (Proposed) |
|---------|--------------|---------------|
| **Pipeline** | Linear (preprocess→model→postprocess) | DAG with branches, loops, merges |
| **Models** | Single model per app | Multiple models with routing |
| **Custom Logic** | JSON rules only | Sandboxed Python processors |
| **Package Format** | Simple ZIP | Signed, versioned, with tests |
| **Distribution** | Manual file sharing | App Store marketplace |
| **Security** | Basic validation | Signing, sandboxing, permissions |
| **Execution** | Sequential | Parallel stage execution |

### Benefits

1. **For Developers**: Create once, distribute globally
2. **For Users**: Browse, install, and run with one click
3. **For Enterprises**: Centralized app management and licensing
4. **For Quality**: Built-in testing and benchmarking

This enhanced architecture transforms MVAS from a simple inference wrapper into a true **Industrial Machine Vision Platform**.

