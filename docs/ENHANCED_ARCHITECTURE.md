# MVAS Enhanced Architecture
## Native Plugin System, Multi-Stage Pipeline & App Marketplace

**Version:** 2.0.0  
**Date:** December 17, 2025  
**Status:** Proposed Enhancement

### Key Features

- 🚀 **Native Plugins (DLL/SO)** - C/C++ compiled plugins for maximum performance
- 📊 **DAG Pipelines** - Multi-stage processing with branching and loops
- 🏪 **App Marketplace** - Distribute and monetize vision applications
- ⚡ **Direct Function Calls** - No HTTP overhead, ~100ms load time
- 🎮 **Shared GPU Memory** - Efficient memory pooling across plugins

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
│  │ • ModelNode   │  │ • Tensors     │  │ • If/Else     │  │ • Native DLL  │   │
│  │ • ProcessNode │  │ • Metadata    │  │ • Switch      │  │ • .SO Plugin  │   │
│  │ • BranchNode  │  │ • Results     │  │ • Loop        │  │ • Python Ext  │   │
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

## 4. Native Plugin Application Architecture (DLL/SO)

### 4.1 Why Native Plugins?

For maximum performance with minimal overhead, applications are compiled as native shared libraries:

| Aspect | Script-Based | Docker | Native DLL/SO |
|--------|--------------|--------|---------------|
| **Load time** | ~2-5s | ~20-100s | **~10-100ms** |
| **Call overhead** | Python GIL | HTTP/gRPC | **Direct call** |
| **Memory** | Python overhead | Container overhead | **Minimal** |
| **GPU sharing** | Limited | Isolated | **Shared pool** |
| **Package size** | 10-500 MB | 2-8 GB | **1-100 MB** |
| **Performance** | Good | Best | **Best** |
| **Flexibility** | Limited | Full | **Full** |

### 4.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         MVAS NATIVE PLUGIN RUNTIME                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  MVAS Runtime Process (Single Process)                                         │
│  ┌───────────────────────────────────────────────────────────────────────────┐ │
│  │                          Plugin Manager                                    │ │
│  │                                                                           │ │
│  │  LoadLibrary() / dlopen()                                                 │ │
│  │         │                                                                 │ │
│  │         ▼                                                                 │ │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐  │ │
│  │  │  App A.dll  │   │  App B.dll  │   │  App C.dll  │   │  App D.dll  │  │ │
│  │  │             │   │             │   │             │   │             │  │ │
│  │  │ TensorRT    │   │ OpenVINO    │   │ Custom C++  │   │ ONNX        │  │ │
│  │  │ PatchCore   │   │ YOLO        │   │ Pipeline    │   │ Segmentation│  │ │
│  │  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘  │ │
│  │         │                 │                 │                 │         │ │
│  │         └─────────────────┴─────────────────┴─────────────────┘         │ │
│  │                                    │                                      │ │
│  │                      Direct Function Calls (C ABI)                        │ │
│  │                           mvas_infer(image, result)                       │ │
│  │                                    │                                      │ │
│  └────────────────────────────────────┼──────────────────────────────────────┘ │
│                                       │                                        │
│  ┌────────────────────────────────────┴──────────────────────────────────────┐ │
│  │                         Shared GPU Memory Pool                             │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────────────────────────┐  │ │
│  │  │ Model A │  │ Model B │  │ Model C │  │      Available Memory       │  │ │
│  │  │  1.5GB  │  │  2.0GB  │  │  1.0GB  │  │          3.5GB              │  │ │
│  │  └─────────┘  └─────────┘  └─────────┘  └─────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────────┘ │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 Plugin Interface Standard (C ABI)

All MVAS application plugins must implement this standard C interface:

```c
// mvas_plugin.h - Standard MVAS Plugin Interface

#ifndef MVAS_PLUGIN_H
#define MVAS_PLUGIN_H

#include <stdint.h>
#include <stdbool.h>

#ifdef _WIN32
    #define MVAS_EXPORT __declspec(dllexport)
#else
    #define MVAS_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Data Structures
// ============================================================================

// Image input structure
typedef struct {
    uint8_t* data;          // Raw pixel data (RGB or BGR)
    int32_t width;          // Image width
    int32_t height;         // Image height  
    int32_t channels;       // Number of channels (3 for RGB)
    int32_t stride;         // Row stride in bytes
    const char* format;     // "RGB", "BGR", "GRAY"
} MVASImage;

// Inference result structure
typedef struct {
    const char* decision;           // "pass", "fail", "review"
    float confidence;               // 0.0 - 1.0
    float anomaly_score;            // 0.0 - 1.0
    float inference_time_ms;        // Inference time
    
    // Optional: Anomaly heatmap
    float* anomaly_map;             // NULL or width*height float array
    int32_t anomaly_map_width;
    int32_t anomaly_map_height;
    
    // Optional: Bounding boxes
    float* bboxes;                  // NULL or [x1,y1,x2,y2,score,class] array
    int32_t num_bboxes;
    
    // Optional: Visualization
    uint8_t* visualization;         // NULL or RGB image data
    int32_t viz_width;
    int32_t viz_height;
    
    // Optional: Custom JSON details
    const char* details_json;       // NULL or JSON string
} MVASResult;

// Plugin information
typedef struct {
    const char* name;               // "Bottle Cap Inspection"
    const char* version;            // "1.0.0"
    const char* author;             // "Company Name"
    const char* description;        // "Detects defects..."
    const char* model_type;         // "anomaly_detection", "classification", etc.
    int32_t input_width;            // Expected input width
    int32_t input_height;           // Expected input height
    const char* input_format;       // "RGB" or "BGR"
} MVASPluginInfo;

// Configuration parameter
typedef struct {
    const char* key;                // "threshold"
    const char* value;              // "0.5"
} MVASConfigParam;

// ============================================================================
// Required Functions (Must be implemented by plugin)
// ============================================================================

// Get plugin information
MVAS_EXPORT const MVASPluginInfo* mvas_get_info(void);

// Initialize plugin (load model, allocate resources)
// Returns: 0 on success, error code on failure
MVAS_EXPORT int32_t mvas_init(const char* model_path, const char* config_json);

// Run inference on single image
// Returns: 0 on success, error code on failure
MVAS_EXPORT int32_t mvas_infer(const MVASImage* image, MVASResult* result);

// Cleanup and release resources
MVAS_EXPORT void mvas_cleanup(void);

// ============================================================================
// Optional Functions
// ============================================================================

// Batch inference
MVAS_EXPORT int32_t mvas_infer_batch(
    const MVASImage* images, 
    int32_t num_images,
    MVASResult* results
);

// Update configuration at runtime
MVAS_EXPORT int32_t mvas_set_config(
    const MVASConfigParam* params, 
    int32_t num_params
);

// Get current configuration
MVAS_EXPORT const char* mvas_get_config(void);

// Warmup (run dummy inference to initialize GPU)
MVAS_EXPORT int32_t mvas_warmup(int32_t iterations);

// Free result memory (if plugin allocates memory)
MVAS_EXPORT void mvas_free_result(MVASResult* result);

// Get last error message
MVAS_EXPORT const char* mvas_get_error(void);

#ifdef __cplusplus
}
#endif

#endif // MVAS_PLUGIN_H
```

### 4.4 .mvapp v2 Package Structure (Native Plugin)

```
bottle_inspection.mvapp (ZIP Archive)
│
├── manifest.json              # App metadata & plugin config (REQUIRED)
│
├── bin/                       # Compiled binaries (REQUIRED)
│   ├── windows/
│   │   └── plugin.dll         # Windows x64 binary
│   ├── linux/
│   │   └── plugin.so          # Linux x64 binary  
│   └── linux-arm64/
│       └── plugin.so          # Linux ARM64 (Jetson)
│
├── models/                    # Model files (REQUIRED)
│   ├── model.engine           # TensorRT engine (platform-specific)
│   ├── model.onnx             # ONNX model (portable)
│   └── model_config.json      # Model configuration
│
├── deps/                      # Dependencies (optional)
│   ├── windows/
│   │   ├── onnxruntime.dll
│   │   └── tensorrt.dll
│   └── linux/
│       ├── libonnxruntime.so
│       └── libnvinfer.so
│
├── config/                    # Configuration files
│   ├── default_params.json    # Default inference parameters
│   └── thresholds.json        # Decision thresholds
│
├── assets/                    # UI & documentation
│   ├── icon.png               # App icon (256x256)
│   ├── preview.png            # Screenshot
│   └── README.md              # Documentation
│
├── test/                      # Test resources
│   ├── test_images/
│   ├── expected_results.json
│   └── benchmark.json
│
└── signature.json             # Digital signature
```

### 4.5 Example Plugin Implementation (C++)

```cpp
// bottle_inspection_plugin.cpp
// Example MVAS Plugin using TensorRT

#include "mvas_plugin.h"
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <string>

using namespace nvinfer1;

// ============================================================================
// Plugin State
// ============================================================================

static struct {
    std::unique_ptr<IRuntime> runtime;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
    
    void* gpu_input;
    void* gpu_output;
    float* cpu_output;
    
    int input_size;
    int output_size;
    
    float threshold;
    bool initialized;
    std::string last_error;
    
    MVASPluginInfo info;
} g_plugin;

// ============================================================================
// Plugin Information
// ============================================================================

MVAS_EXPORT const MVASPluginInfo* mvas_get_info(void) {
    g_plugin.info.name = "Bottle Cap Inspection";
    g_plugin.info.version = "2.0.0";
    g_plugin.info.author = "Vision Systems Inc.";
    g_plugin.info.description = "High-performance bottle cap defect detection using TensorRT";
    g_plugin.info.model_type = "anomaly_detection";
    g_plugin.info.input_width = 256;
    g_plugin.info.input_height = 256;
    g_plugin.info.input_format = "RGB";
    return &g_plugin.info;
}

// ============================================================================
// Initialization
// ============================================================================

MVAS_EXPORT int32_t mvas_init(const char* model_path, const char* config_json) {
    try {
        // Parse config
        if (config_json) {
            // Parse JSON config for threshold, etc.
            g_plugin.threshold = 0.5f; // Default
        }
        
        // Load TensorRT engine
        std::ifstream file(model_path, std::ios::binary);
        if (!file) {
            g_plugin.last_error = "Failed to open model file";
            return -1;
        }
        
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> engine_data(size);
        file.read(engine_data.data(), size);
        
        // Create runtime and engine
        g_plugin.runtime.reset(createInferRuntime(gLogger));
        g_plugin.engine.reset(
            g_plugin.runtime->deserializeCudaEngine(engine_data.data(), size)
        );
        g_plugin.context.reset(g_plugin.engine->createExecutionContext());
        
        // Allocate GPU memory
        auto input_dims = g_plugin.engine->getBindingDimensions(0);
        auto output_dims = g_plugin.engine->getBindingDimensions(1);
        
        g_plugin.input_size = 1 * 3 * 256 * 256 * sizeof(float);
        g_plugin.output_size = 1 * 256 * 256 * sizeof(float);
        
        cudaMalloc(&g_plugin.gpu_input, g_plugin.input_size);
        cudaMalloc(&g_plugin.gpu_output, g_plugin.output_size);
        g_plugin.cpu_output = new float[256 * 256];
        
        g_plugin.initialized = true;
        return 0;
        
    } catch (const std::exception& e) {
        g_plugin.last_error = e.what();
        return -1;
    }
}

// ============================================================================
// Inference
// ============================================================================

MVAS_EXPORT int32_t mvas_infer(const MVASImage* image, MVASResult* result) {
    if (!g_plugin.initialized) {
        g_plugin.last_error = "Plugin not initialized";
        return -1;
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        // Preprocess image
        cv::Mat img(image->height, image->width, CV_8UC3, image->data);
        cv::Mat resized, normalized;
        
        cv::resize(img, resized, cv::Size(256, 256));
        resized.convertTo(normalized, CV_32FC3, 1.0/255.0);
        
        // Apply ImageNet normalization
        cv::subtract(normalized, cv::Scalar(0.485, 0.456, 0.406), normalized);
        cv::divide(normalized, cv::Scalar(0.229, 0.224, 0.225), normalized);
        
        // Convert HWC to CHW
        std::vector<float> input_data(3 * 256 * 256);
        std::vector<cv::Mat> channels(3);
        cv::split(normalized, channels);
        for (int c = 0; c < 3; c++) {
            memcpy(input_data.data() + c * 256 * 256, 
                   channels[c].data, 
                   256 * 256 * sizeof(float));
        }
        
        // Copy to GPU
        cudaMemcpy(g_plugin.gpu_input, input_data.data(), 
                   g_plugin.input_size, cudaMemcpyHostToDevice);
        
        // Run inference
        void* bindings[] = {g_plugin.gpu_input, g_plugin.gpu_output};
        g_plugin.context->executeV2(bindings);
        
        // Copy result back
        cudaMemcpy(g_plugin.cpu_output, g_plugin.gpu_output,
                   g_plugin.output_size, cudaMemcpyDeviceToHost);
        
        // Calculate anomaly score (max of anomaly map)
        float max_score = 0.0f;
        for (int i = 0; i < 256 * 256; i++) {
            if (g_plugin.cpu_output[i] > max_score) {
                max_score = g_plugin.cpu_output[i];
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        float inference_ms = std::chrono::duration<float, std::milli>(end - start).count();
        
        // Fill result
        result->anomaly_score = max_score;
        result->inference_time_ms = inference_ms;
        
        if (max_score < g_plugin.threshold * 0.6f) {
            result->decision = "pass";
            result->confidence = 1.0f - max_score;
        } else if (max_score < g_plugin.threshold) {
            result->decision = "review";
            result->confidence = max_score;
        } else {
            result->decision = "fail";
            result->confidence = max_score;
        }
        
        // Provide anomaly map
        result->anomaly_map = g_plugin.cpu_output;
        result->anomaly_map_width = 256;
        result->anomaly_map_height = 256;
        
        return 0;
        
    } catch (const std::exception& e) {
        g_plugin.last_error = e.what();
        return -1;
    }
}

// ============================================================================
// Configuration
// ============================================================================

MVAS_EXPORT int32_t mvas_set_config(const MVASConfigParam* params, int32_t num_params) {
    for (int i = 0; i < num_params; i++) {
        if (strcmp(params[i].key, "threshold") == 0) {
            g_plugin.threshold = atof(params[i].value);
        }
    }
    return 0;
}

MVAS_EXPORT int32_t mvas_warmup(int32_t iterations) {
    // Create dummy image
    MVASImage dummy;
    std::vector<uint8_t> dummy_data(256 * 256 * 3, 128);
    dummy.data = dummy_data.data();
    dummy.width = 256;
    dummy.height = 256;
    dummy.channels = 3;
    dummy.stride = 256 * 3;
    dummy.format = "RGB";
    
    MVASResult result;
    for (int i = 0; i < iterations; i++) {
        mvas_infer(&dummy, &result);
    }
    return 0;
}

// ============================================================================
// Cleanup
// ============================================================================

MVAS_EXPORT void mvas_cleanup(void) {
    if (g_plugin.gpu_input) cudaFree(g_plugin.gpu_input);
    if (g_plugin.gpu_output) cudaFree(g_plugin.gpu_output);
    if (g_plugin.cpu_output) delete[] g_plugin.cpu_output;
    
    g_plugin.context.reset();
    g_plugin.engine.reset();
    g_plugin.runtime.reset();
    g_plugin.initialized = false;
}

MVAS_EXPORT const char* mvas_get_error(void) {
    return g_plugin.last_error.c_str();
}

MVAS_EXPORT void mvas_free_result(MVASResult* result) {
    // Anomaly map points to internal buffer, no need to free
    // Only free if we allocated visualization
    if (result->visualization) {
        delete[] result->visualization;
        result->visualization = nullptr;
    }
}
```

### 4.6 Manifest Schema (manifest.json)

```json
{
  "$schema": "https://mvas.io/schema/manifest-v2-native.json",
  "manifest_version": "2.0.0",
  "package_type": "native",
  
  "app": {
    "id": "com.company.bottle-inspection",
    "name": "Bottle Cap Inspection (TensorRT)",
    "version": "2.1.0",
    "version_code": 21,
    "description": "High-performance bottle cap defect detection using TensorRT",
    "author": {
      "name": "Vision Systems Inc.",
      "email": "support@visionsystems.com",
      "website": "https://visionsystems.com"
    },
    "license": "commercial",
    "category": "anomaly_detection",
    "tags": ["bottle", "cap", "anomaly", "tensorrt", "manufacturing"],
    "icon": "assets/icon.png",
    "preview_images": ["assets/preview.png"],
    "documentation": "assets/README.md"
  },
  
  "plugin": {
    "interface_version": "2.0",
    "binaries": {
      "windows-x64": "bin/windows/plugin.dll",
      "linux-x64": "bin/linux/plugin.so",
      "linux-arm64": "bin/linux-arm64/plugin.so"
    },
    "model": {
      "path": "models/model.engine",
      "fallback": "models/model.onnx",
      "config": "models/model_config.json"
    },
    "dependencies": {
      "windows-x64": ["deps/windows/onnxruntime.dll"],
      "linux-x64": ["deps/linux/libonnxruntime.so"]
    }
  },
  
  "requirements": {
    "cuda_version": ">=11.8",
    "tensorrt_version": ">=8.5",
    "gpu_memory_mb": 2048,
    "system_memory_mb": 4096,
    "gpu_required": true,
    "gpu_compute_capability": ">=7.5"
  },
  
  "interface": {
    "input": {
      "type": "image",
      "formats": ["jpeg", "png", "bmp"],
      "color_mode": "RGB",
      "resolution": {
        "width": 256,
        "height": 256,
        "resize_mode": "stretch"
      },
      "max_batch_size": 8
    },
    "output": {
      "type": "anomaly_detection",
      "provides": [
        "decision",
        "anomaly_score", 
        "confidence",
        "anomaly_map",
        "visualization"
      ]
    }
  },
  
  "configuration": {
    "default_config": "config/default_params.json",
    "user_configurable": [
      {
        "key": "threshold",
        "type": "float",
        "default": 0.5,
        "min": 0.0,
        "max": 1.0,
        "label": "Anomaly Threshold"
      },
      {
        "key": "visualization_type",
        "type": "enum",
        "default": "heatmap",
        "options": ["heatmap", "contour", "bbox", "none"],
        "label": "Visualization Type"
      }
    ]
  },
  
  "performance": {
    "baseline": {
      "load_time_ms": 500,
      "inference_time_ms": 15,
      "throughput_fps": 60
    },
    "hardware_tested": {
      "gpu": "NVIDIA RTX 3080",
      "cpu": "Intel i9-12900K"
    }
  },
  
  "testing": {
    "test_images": "test/test_images/",
    "expected_results": "test/expected_results.json"
  },
  
  "marketplace": {
    "pricing": "subscription",
    "price_monthly_usd": 99.0,
    "trial_days": 14
  }
}
```

### 4.6 CMake Build Template

```cmake
# CMakeLists.txt - MVAS Plugin Build Configuration

cmake_minimum_required(VERSION 3.18)
project(mvas_bottle_inspection VERSION 2.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Find Dependencies
# ============================================================================

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT} /usr/include/x86_64-linux-gnu /usr/local/TensorRT/include)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS ${TENSORRT_ROOT}/lib /usr/lib/x86_64-linux-gnu /usr/local/TensorRT/lib)

# OpenCV (for image preprocessing)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# ============================================================================
# Plugin Library
# ============================================================================

add_library(mvas_plugin SHARED
    src/plugin.cpp
    src/inference_engine.cpp
    src/preprocessing.cpp
)

# Include MVAS SDK header
target_include_directories(mvas_plugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(mvas_plugin PRIVATE
    ${TENSORRT_LIBRARY}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
)

# Export all required symbols
if(WIN32)
    set_target_properties(mvas_plugin PROPERTIES
        WINDOWS_EXPORT_ALL_SYMBOLS ON
        OUTPUT_NAME "plugin"
        SUFFIX ".dll"
    )
else()
    set_target_properties(mvas_plugin PROPERTIES
        OUTPUT_NAME "plugin"
        SUFFIX ".so"
        C_VISIBILITY_PRESET default
        CXX_VISIBILITY_PRESET default
    )
endif()

# ============================================================================
# Install
# ============================================================================

# Platform-specific output directory
if(WIN32)
    set(PLATFORM_DIR "windows")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(PLATFORM_DIR "linux-arm64")
else()
    set(PLATFORM_DIR "linux")
endif()

install(TARGETS mvas_plugin
    LIBRARY DESTINATION bin/${PLATFORM_DIR}
    RUNTIME DESTINATION bin/${PLATFORM_DIR}
)
```

### 4.6.1 Project Structure for Plugin Development

```
my_inspection_plugin/
├── CMakeLists.txt              # Build configuration
├── include/
│   └── mvas_plugin.h           # MVAS SDK header (copy from SDK)
├── src/
│   ├── plugin.cpp              # Main plugin implementation
│   ├── inference_engine.cpp    # TensorRT inference
│   ├── inference_engine.h
│   ├── preprocessing.cpp       # Image preprocessing
│   └── preprocessing.h
├── models/
│   ├── model.onnx              # Original ONNX model
│   └── model_config.json       # Model configuration
├── config/
│   └── default_params.json     # Default parameters
├── scripts/
│   ├── build_engine.py         # Convert ONNX to TensorRT
│   └── build.sh                # Build script
└── test/
    ├── test_images/
    └── test_plugin.cpp         # Unit tests
```

### 4.6.2 Build Scripts

```bash
#!/bin/bash
# scripts/build.sh - Build plugin for current platform

set -e

BUILD_DIR="build"
INSTALL_DIR="../package"

# Clean and create build directory
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTENSORRT_ROOT=/usr/local/TensorRT \
    -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR

# Build
cmake --build . --config Release -j$(nproc)

# Install
cmake --install .

echo "Build complete! Plugin installed to $INSTALL_DIR"
```

```powershell
# scripts/build.ps1 - Build plugin for Windows

$BuildDir = "build"
$InstallDir = "../package"

# Clean and create build directory
if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
New-Item -ItemType Directory -Path $BuildDir

Push-Location $BuildDir

# Configure with Visual Studio
cmake .. `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DTENSORRT_ROOT="C:/TensorRT" `
    -DCMAKE_INSTALL_PREFIX="$InstallDir"

# Build
cmake --build . --config Release

# Install
cmake --install . --config Release

Pop-Location

Write-Host "Build complete! Plugin installed to $InstallDir"
```

### 4.6.3 Cross-Platform Build with GitHub Actions

```yaml
# .github/workflows/build.yml
name: Build MVAS Plugin

on:
  push:
    branches: [main]
  release:
    types: [created]

jobs:
  build-windows:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup CUDA
        uses: Jimver/cuda-toolkit@v0.2.11
        with:
          cuda: '11.8.0'
      
      - name: Install TensorRT
        run: |
          # Download and install TensorRT
          choco install tensorrt -y
      
      - name: Build
        run: |
          mkdir build && cd build
          cmake .. -G "Visual Studio 17 2022" -A x64
          cmake --build . --config Release
      
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: plugin-windows
          path: build/Release/plugin.dll

  build-linux:
    runs-on: ubuntu-latest
    container: nvcr.io/nvidia/tensorrt:23.10-py3
    steps:
      - uses: actions/checkout@v3
      
      - name: Install dependencies
        run: |
          apt-get update
          apt-get install -y cmake libopencv-dev
      
      - name: Build
        run: |
          mkdir build && cd build
          cmake .. -DCMAKE_BUILD_TYPE=Release
          cmake --build . -j$(nproc)
      
      - name: Upload artifact
        uses: actions/upload-artifact@v3
        with:
          name: plugin-linux
          path: build/plugin.so

  package:
    needs: [build-windows, build-linux]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Download artifacts
        uses: actions/download-artifact@v3
      
      - name: Create .mvapp package
        run: |
          mkdir -p package/bin/windows package/bin/linux
          cp plugin-windows/plugin.dll package/bin/windows/
          cp plugin-linux/plugin.so package/bin/linux/
          cp -r models config assets package/
          cp manifest.json package/
          cd package && zip -r ../my-plugin.mvapp .
      
      - name: Upload .mvapp
        uses: actions/upload-artifact@v3
        with:
          name: mvapp-package
          path: my-plugin.mvapp
```

### 4.7 MVAS Runtime Plugin Loader (Python/C++)

The MVAS runtime loads and manages plugins dynamically:

```python
# mvas_runtime/plugin_loader.py
import ctypes
import platform
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np

# ============================================================================
# C Structure Mappings
# ============================================================================

class MVASImage(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("channels", ctypes.c_int32),
        ("stride", ctypes.c_int32),
        ("format", ctypes.c_char_p),
    ]

class MVASResult(ctypes.Structure):
    _fields_ = [
        ("decision", ctypes.c_char_p),
        ("confidence", ctypes.c_float),
        ("anomaly_score", ctypes.c_float),
        ("inference_time_ms", ctypes.c_float),
        ("anomaly_map", ctypes.POINTER(ctypes.c_float)),
        ("anomaly_map_width", ctypes.c_int32),
        ("anomaly_map_height", ctypes.c_int32),
        ("bboxes", ctypes.POINTER(ctypes.c_float)),
        ("num_bboxes", ctypes.c_int32),
        ("visualization", ctypes.POINTER(ctypes.c_uint8)),
        ("viz_width", ctypes.c_int32),
        ("viz_height", ctypes.c_int32),
        ("details_json", ctypes.c_char_p),
    ]

class MVASPluginInfo(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("version", ctypes.c_char_p),
        ("author", ctypes.c_char_p),
        ("description", ctypes.c_char_p),
        ("model_type", ctypes.c_char_p),
        ("input_width", ctypes.c_int32),
        ("input_height", ctypes.c_int32),
        ("input_format", ctypes.c_char_p),
    ]

# ============================================================================
# Plugin Wrapper Class
# ============================================================================

@dataclass
class PluginInfo:
    name: str
    version: str
    author: str
    description: str
    model_type: str
    input_width: int
    input_height: int
    input_format: str

@dataclass 
class InferenceResult:
    decision: str
    confidence: float
    anomaly_score: float
    inference_time_ms: float
    anomaly_map: Optional[np.ndarray] = None
    bboxes: Optional[np.ndarray] = None
    visualization: Optional[np.ndarray] = None
    details: Optional[dict] = None


class MVASPlugin:
    """Wrapper for MVAS native plugin (DLL/SO)"""
    
    def __init__(self, plugin_path: Path, model_path: Path, config_json: str = "{}"):
        self.plugin_path = plugin_path
        self.model_path = model_path
        self._lib = None
        self._loaded = False
        
        # Load library
        self._load_library()
        self._setup_function_signatures()
        
        # Initialize plugin
        result = self._lib.mvas_init(
            str(model_path).encode('utf-8'),
            config_json.encode('utf-8')
        )
        if result != 0:
            error = self._lib.mvas_get_error()
            raise RuntimeError(f"Plugin init failed: {error.decode()}")
        
        self._loaded = True
    
    def _load_library(self):
        """Load the native library"""
        system = platform.system()
        
        if system == "Windows":
            self._lib = ctypes.CDLL(str(self.plugin_path))
        else:  # Linux/Mac
            self._lib = ctypes.CDLL(str(self.plugin_path), mode=ctypes.RTLD_GLOBAL)
    
    def _setup_function_signatures(self):
        """Set up C function signatures"""
        # mvas_get_info
        self._lib.mvas_get_info.argtypes = []
        self._lib.mvas_get_info.restype = ctypes.POINTER(MVASPluginInfo)
        
        # mvas_init
        self._lib.mvas_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._lib.mvas_init.restype = ctypes.c_int32
        
        # mvas_infer
        self._lib.mvas_infer.argtypes = [
            ctypes.POINTER(MVASImage),
            ctypes.POINTER(MVASResult)
        ]
        self._lib.mvas_infer.restype = ctypes.c_int32
        
        # mvas_cleanup
        self._lib.mvas_cleanup.argtypes = []
        self._lib.mvas_cleanup.restype = None
        
        # mvas_get_error
        self._lib.mvas_get_error.argtypes = []
        self._lib.mvas_get_error.restype = ctypes.c_char_p
        
        # mvas_warmup (optional)
        try:
            self._lib.mvas_warmup.argtypes = [ctypes.c_int32]
            self._lib.mvas_warmup.restype = ctypes.c_int32
            self._has_warmup = True
        except AttributeError:
            self._has_warmup = False
    
    def get_info(self) -> PluginInfo:
        """Get plugin information"""
        info_ptr = self._lib.mvas_get_info()
        info = info_ptr.contents
        return PluginInfo(
            name=info.name.decode() if info.name else "",
            version=info.version.decode() if info.version else "",
            author=info.author.decode() if info.author else "",
            description=info.description.decode() if info.description else "",
            model_type=info.model_type.decode() if info.model_type else "",
            input_width=info.input_width,
            input_height=info.input_height,
            input_format=info.input_format.decode() if info.input_format else "RGB"
        )
    
    def infer(self, image: np.ndarray) -> InferenceResult:
        """Run inference on image (HWC uint8 RGB/BGR array)"""
        if not self._loaded:
            raise RuntimeError("Plugin not initialized")
        
        # Prepare image structure
        h, w, c = image.shape
        img_data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        
        mvas_image = MVASImage()
        mvas_image.data = img_data
        mvas_image.width = w
        mvas_image.height = h
        mvas_image.channels = c
        mvas_image.stride = w * c
        mvas_image.format = b"RGB"
        
        # Prepare result structure
        result = MVASResult()
        
        # Call inference
        ret = self._lib.mvas_infer(ctypes.byref(mvas_image), ctypes.byref(result))
        
        if ret != 0:
            error = self._lib.mvas_get_error()
            raise RuntimeError(f"Inference failed: {error.decode()}")
        
        # Extract results
        inference_result = InferenceResult(
            decision=result.decision.decode() if result.decision else "unknown",
            confidence=result.confidence,
            anomaly_score=result.anomaly_score,
            inference_time_ms=result.inference_time_ms
        )
        
        # Extract anomaly map if available
        if result.anomaly_map and result.anomaly_map_width > 0:
            map_size = result.anomaly_map_width * result.anomaly_map_height
            inference_result.anomaly_map = np.ctypeslib.as_array(
                result.anomaly_map, shape=(result.anomaly_map_height, result.anomaly_map_width)
            ).copy()
        
        return inference_result
    
    def warmup(self, iterations: int = 10):
        """Warm up the model"""
        if self._has_warmup:
            self._lib.mvas_warmup(iterations)
    
    def cleanup(self):
        """Release resources"""
        if self._loaded:
            self._lib.mvas_cleanup()
            self._loaded = False
    
    def __del__(self):
        self.cleanup()


# ============================================================================
# Plugin Manager
# ============================================================================

class PluginManager:
    """Manages multiple loaded plugins"""
    
    def __init__(self):
        self.plugins: dict[str, MVASPlugin] = {}
    
    def load_plugin(self, app_id: str, plugin_path: Path, model_path: Path, 
                   config_json: str = "{}") -> MVASPlugin:
        """Load a plugin"""
        if app_id in self.plugins:
            self.unload_plugin(app_id)
        
        plugin = MVASPlugin(plugin_path, model_path, config_json)
        self.plugins[app_id] = plugin
        return plugin
    
    def get_plugin(self, app_id: str) -> Optional[MVASPlugin]:
        """Get a loaded plugin"""
        return self.plugins.get(app_id)
    
    def unload_plugin(self, app_id: str):
        """Unload a plugin"""
        if app_id in self.plugins:
            self.plugins[app_id].cleanup()
            del self.plugins[app_id]
    
    def unload_all(self):
        """Unload all plugins"""
        for app_id in list(self.plugins.keys()):
            self.unload_plugin(app_id)
```

### 4.8 Multi-Stage Pipeline with Plugins

For complex DAG pipelines, chain multiple plugins:

```python
# mvas_runtime/plugin_pipeline.py
from typing import List, Dict, Any
import numpy as np
from .plugin_loader import PluginManager, MVASPlugin

class PluginPipeline:
    """Execute multi-stage inference with multiple plugins"""
    
    def __init__(self, plugin_manager: PluginManager):
        self.manager = plugin_manager
        self.stages: List[Dict[str, Any]] = []
    
    def add_stage(self, stage_id: str, app_id: str, config: Dict = None):
        """Add a processing stage"""
        plugin = self.manager.get_plugin(app_id)
        if not plugin:
            raise ValueError(f"Plugin {app_id} not loaded")
        
        self.stages.append({
            "id": stage_id,
            "app_id": app_id,
            "plugin": plugin,
            "config": config or {}
        })
    
    def execute(self, image: np.ndarray) -> Dict[str, Any]:
        """Execute full pipeline"""
        results = {}
        current_data = {"image": image}
        
        for stage in self.stages:
            stage_id = stage["id"]
            plugin = stage["plugin"]
            
            # Run inference
            result = plugin.infer(current_data["image"])
            results[stage_id] = {
                "decision": result.decision,
                "confidence": result.confidence,
                "anomaly_score": result.anomaly_score,
                "inference_time_ms": result.inference_time_ms
            }
            
            # Pass data to next stage
            if result.anomaly_map is not None:
                current_data["anomaly_map"] = result.anomaly_map
        
        # Aggregate final result
        final_decision = "pass"
        for stage_result in results.values():
            if stage_result["decision"] == "fail":
                final_decision = "fail"
                break
            elif stage_result["decision"] == "review":
                final_decision = "review"
        
        return {
            "final_decision": final_decision,
            "stage_results": results,
            "total_time_ms": sum(r["inference_time_ms"] for r in results.values())
        }


# Example usage
if __name__ == "__main__":
    manager = PluginManager()
    
    # Load plugins
    manager.load_plugin(
        "detector",
        Path("apps/detector/bin/windows/plugin.dll"),
        Path("apps/detector/models/detector.engine")
    )
    manager.load_plugin(
        "classifier",
        Path("apps/classifier/bin/windows/plugin.dll"),
        Path("apps/classifier/models/classifier.engine")
    )
    manager.load_plugin(
        "anomaly",
        Path("apps/anomaly/bin/windows/plugin.dll"),
        Path("apps/anomaly/models/anomaly.engine")
    )
    
    # Create pipeline
    pipeline = PluginPipeline(manager)
    pipeline.add_stage("detection", "detector")
    pipeline.add_stage("classification", "classifier")
    pipeline.add_stage("anomaly_detection", "anomaly")
    
    # Run inference
    import cv2
    image = cv2.imread("test.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    result = pipeline.execute(image)
    print(f"Decision: {result['final_decision']}")
    print(f"Total time: {result['total_time_ms']:.1f}ms")
```

---

## 4B. Alternative: Lightweight Script-Based Package

For simpler applications that don't need custom environments, the original script-based approach is still supported:

```
simple_app.mvapp (ZIP Archive)
│
├── manifest.json              # App metadata (package_type: "script")
├── model/
│   └── model.onnx
├── preprocessing/
│   └── transforms.json
├── postprocessing/
│   └── rules.json
└── assets/
    └── icon.png
```

The runtime auto-detects the package type from `manifest.json`:

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

## 5. MVAS Native Plugin Runtime

### 5.1 Plugin Lifecycle Management

```python
"""
MVAS Native Plugin Manager
Handles DLL/SO plugin lifecycle for vision applications
"""

import ctypes
import platform
import json
import zipfile
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np

logger = logging.getLogger(__name__)

class PluginState(Enum):
    LOADING = "loading"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"

@dataclass
class PluginInfo:
    app_id: str
    name: str
    version: str
    state: PluginState
    plugin_path: Path
    model_path: Path
    gpu_id: Optional[int]
    load_time_ms: float
    avg_inference_ms: float = 0.0

class MVASPluginManager:
    """
    Manages native DLL/SO plugins for MVAS applications.
    
    Features:
    - Load plugins from .mvapp packages
    - Direct C function calls (no network overhead)
    - Shared GPU memory pool
    - Hot-reload support
    - Performance monitoring
    """
    
    def __init__(self, plugins_dir: Path = Path("plugins")):
        self.plugins_dir = plugins_dir
        self.plugins: Dict[str, 'NativePlugin'] = {}
        self.plugin_info: Dict[str, PluginInfo] = {}
        self._gpu_allocator = GPUMemoryAllocator()
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
    def load_app(self, mvapp_path: str) -> PluginInfo:
        """Load a plugin from .mvapp package"""
        import time
        start_time = time.perf_counter()
        
        mvapp_path = Path(mvapp_path)
        
        with zipfile.ZipFile(mvapp_path, 'r') as zf:
            # Read manifest
            manifest = json.loads(zf.read("manifest.json"))
            app_id = manifest["app"]["id"]
            plugin_config = manifest["plugin"]
            
            # Determine platform
            system = platform.system().lower()
            arch = "arm64" if platform.machine() == "aarch64" else "x64"
            platform_key = f"{system}-{arch}" if system != "windows" else "windows-x64"
            
            # Get binary path
            binary_rel_path = plugin_config["binaries"].get(platform_key)
            if not binary_rel_path:
                raise RuntimeError(f"No binary for platform: {platform_key}")
            
            # Extract plugin to plugins directory
            app_dir = self.plugins_dir / app_id
            app_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract all files
            zf.extractall(app_dir)
            
            plugin_path = app_dir / binary_rel_path
            model_path = app_dir / plugin_config.get("model", {}).get("path", "models/model.onnx")
            config_json = json.dumps(manifest.get("configuration", {}).get("default_config", {}))
            
            # Load native plugin
            logger.info(f"Loading plugin: {plugin_path}")
            plugin = NativePlugin(plugin_path, model_path, config_json)
            
            self.plugins[app_id] = plugin
            
            load_time = (time.perf_counter() - start_time) * 1000
            
            # Get plugin info
            native_info = plugin.get_info()
            
            info = PluginInfo(
                app_id=app_id,
                name=native_info.name,
                version=native_info.version,
                state=PluginState.READY,
                plugin_path=plugin_path,
                model_path=model_path,
                gpu_id=0,  # Will be allocated
                load_time_ms=load_time
            )
            
            self.plugin_info[app_id] = info
            logger.info(f"Plugin loaded in {load_time:.1f}ms: {native_info.name} v{native_info.version}")
            
            return info
    
    def infer(
        self, 
        app_id: str, 
        image: np.ndarray,
        params: dict = None
    ) -> dict:
        """Run inference using native plugin (direct function call)"""
        if app_id not in self.plugins:
            raise ValueError(f"App not loaded: {app_id}")
        
        plugin = self.plugins[app_id]
        info = self.plugin_info[app_id]
        
        if info.state != PluginState.READY:
            raise RuntimeError(f"Plugin not ready: {info.state}")
        
        # Apply configuration if provided
        if params:
            plugin.set_config(params)
        
        # Direct function call - no network overhead!
        result = plugin.infer(image)
        
        # Update average inference time
        info.avg_inference_ms = (info.avg_inference_ms + result.inference_time_ms) / 2
        
        return {
            "decision": result.decision,
            "confidence": result.confidence,
            "anomaly_score": result.anomaly_score,
            "inference_time_ms": result.inference_time_ms,
            "anomaly_map": result.anomaly_map.tolist() if result.anomaly_map is not None else None
        }
    
    def unload_app(self, app_id: str):
        """Unload plugin and release resources"""
        if app_id not in self.plugins:
            return
        
        try:
            self.plugins[app_id].cleanup()
        except Exception as e:
            logger.warning(f"Error unloading plugin: {e}")
        
        del self.plugins[app_id]
        del self.plugin_info[app_id]
        logger.info(f"Plugin unloaded: {app_id}")
    
    def list_apps(self) -> List[PluginInfo]:
        """List all loaded plugins"""
        return list(self.plugin_info.values())
    
    def warmup(self, app_id: str, iterations: int = 10):
        """Warm up plugin (pre-load GPU kernels)"""
        if app_id in self.plugins:
            self.plugins[app_id].warmup(iterations)
    
    def get_stats(self, app_id: str) -> dict:
        """Get plugin statistics"""
        if app_id not in self.plugin_info:
            raise ValueError(f"App not loaded: {app_id}")
        
        info = self.plugin_info[app_id]
        return {
            "app_id": info.app_id,
            "name": info.name,
            "version": info.version,
            "state": info.state.value,
            "load_time_ms": info.load_time_ms,
            "avg_inference_ms": info.avg_inference_ms
        }
```

### 5.2 CLI Commands for Plugin Management

```bash
# Package a native plugin application
mvas package ./my_plugin_dir -o my_app.mvapp

# Load an application  
mvas load my_app.mvapp

# List loaded plugins
mvas list
# Output:
# ID                    STATUS   LOAD_TIME   AVG_INFER   VERSION
# bottle-inspection     READY    45.2ms      12.5ms      2.1.0
# pcb-detector          READY    62.1ms      18.3ms      1.0.0

# Run inference (direct function call - ultra fast!)
mvas infer bottle-inspection image.jpg --visualize

# Run batch inference
mvas infer bottle-inspection ./test_images/ --batch --output results/

# Benchmark plugin performance
mvas benchmark bottle-inspection --iterations 1000
# Output:
# Plugin: Bottle Cap Inspection v2.1.0
# Iterations: 1000
# Avg inference: 12.5ms
# P95 latency: 15.2ms
# P99 latency: 18.1ms
# Throughput: 80 FPS

# Unload an application
mvas unload bottle-inspection

# Warmup plugin (pre-load GPU kernels)
mvas warmup bottle-inspection --iterations 10

# Get plugin info
mvas info bottle-inspection
```

### 5.3 GPU Memory Pool

```python
class GPUMemoryAllocator:
    """
    Manages shared GPU memory pool across all plugins.
    
    Unlike Docker containers which isolate GPU memory,
    native plugins can share a common GPU memory pool
    for maximum efficiency.
    
    Features:
    - Shared CUDA context across plugins
    - Memory pooling for reduced fragmentation
    - Multi-GPU support with smart allocation
    """
    
    def __init__(self):
        self.gpus = self._discover_gpus()
        self.allocations: Dict[str, int] = {}  # app_id -> gpu_id
    
    def _discover_gpus(self) -> List[dict]:
        """Discover available GPUs using nvidia-smi"""
        import subprocess
        import xml.etree.ElementTree as ET
        
        result = subprocess.run(
            ["nvidia-smi", "-q", "-x"],
            capture_output=True,
            text=True
        )
        
        root = ET.fromstring(result.stdout)
        gpus = []
        
        for i, gpu in enumerate(root.findall(".//gpu")):
            memory = gpu.find(".//fb_memory_usage")
            gpus.append({
                "id": i,
                "name": gpu.find("product_name").text,
                "memory_total_mb": int(memory.find("total").text.split()[0]),
                "memory_used_mb": int(memory.find("used").text.split()[0]),
                "memory_free_mb": int(memory.find("free").text.split()[0]),
            })
        
        return gpus
    
    def allocate(self, app_id: str, memory_required_mb: int) -> int:
        """Allocate a GPU for an application"""
        # Find GPU with enough free memory
        for gpu in self.gpus:
            if gpu["memory_free_mb"] >= memory_required_mb:
                self.allocations[app_id] = gpu["id"]
                gpu["memory_free_mb"] -= memory_required_mb
                return gpu["id"]
        
        raise RuntimeError("No GPU available with sufficient memory")
    
    def release(self, app_id: str):
        """Release GPU allocation"""
        if app_id in self.allocations:
            del self.allocations[app_id]
```

---

## 6. MVAS App Marketplace Architecture

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

## 10. Summary

### Key Enhancements

| Feature | v1 (Current) | v2 Native Plugin (Proposed) |
|---------|--------------|---------------------------|
| **Runtime** | Python/ONNX only | Native DLL/SO (any language) |
| **Performance** | Generic inference | TensorRT, custom CUDA, optimized |
| **Load Time** | ~2-5 seconds | **~10-100 milliseconds** |
| **Call Overhead** | Python GIL | **Direct C function call** |
| **Pipeline** | Linear single model | DAG with multiple plugins |
| **Models** | Single ONNX model | Multiple models, any format |
| **GPU Memory** | Separate pools | **Shared GPU memory pool** |
| **Package Size** | 10-500 MB | **1-100 MB** |
| **Dependencies** | Sandboxed Python | Bundled in plugin |
| **Distribution** | Manual file sharing | Marketplace + download |

### Package Type Comparison

| Aspect | Script-Based | Docker | Native Plugin |
|--------|--------------|--------|---------------|
| **Size** | Small (MB) | Large (GB) | **Tiny (MB)** |
| **Startup** | ~2s | 5-30s | **<100ms** |
| **Overhead** | Python | HTTP/gRPC | **None** |
| **GPU Sharing** | Limited | Isolated | **Full** |
| **Flexibility** | Limited | Full | **Full** |
| **Performance** | Good | Best | **Best** |
| **Development** | Easy | Docker needed | C/C++ needed |
| **Use Case** | Prototypes | Cloud/K8s | **Edge/Production** |

### Architecture Benefits

**For Developers:**
- Full control over inference optimization
- Use any framework (TensorRT, OpenVINO, custom CUDA)
- Standard C ABI - build with C, C++, Rust, Go, etc.
- Compile once per platform, run everywhere

**For Users:**
- Instant loading (~100ms vs 30s for containers)
- Maximum inference performance
- Shared GPU memory across plugins
- Small download sizes (MB not GB)

**For Enterprises:**
- Minimal runtime overhead
- Direct function calls (no network latency)
- Easy to profile and optimize
- Works on edge devices (Jetson, embedded)

### Performance Comparison

| Metric | Python Script | Docker Container | Native Plugin |
|--------|---------------|------------------|---------------|
| Load time | 2-5s | 10-60s | **10-100ms** |
| Infer call | 50-100ms | 20-50ms | **15-25ms** |
| Memory per app | 500MB-2GB | 2-8GB | **100-500MB** |
| GPU overhead | High | Medium | **Minimal** |
| Throughput | 10-20 FPS | 30-60 FPS | **60-120 FPS** |

**For Quality:**
- Reproducible environments
- Built-in testing and benchmarks
- Performance baselines
- Automated validation

### Recommended Approach

```
┌─────────────────────────────────────────────────────────────────┐
│                    Decision Guide                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Simple model, quick prototype?                                 │
│  └─▶ Use Script-Based .mvapp (ONNX + Python)                   │
│                                                                 │
│  Production deployment, need performance?                       │
│  └─▶ Use Native Plugin .mvapp (C/C++ DLL/SO)                   │
│                                                                 │
│  Multi-model pipeline with complex logic?                       │
│  └─▶ Chain multiple plugins with DAG pipeline                  │
│                                                                 │
│  Edge deployment (Jetson, limited resources)?                   │
│  └─▶ Use Native Plugin with ARM64 cross-compile               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

This enhanced architecture transforms MVAS from a simple inference wrapper into a true **Industrial Machine Vision Platform** that supports:

1. **High-Performance Inference** - Direct C calls, TensorRT, custom CUDA
2. **Flexible Development** - C, C++, Rust, or any language with C ABI
3. **Production Ready** - Shared GPU memory, ~100ms load time, 60+ FPS
4. **Easy Distribution** - Small packages (MB not GB), marketplace ecosystem

