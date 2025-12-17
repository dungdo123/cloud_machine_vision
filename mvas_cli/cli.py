"""
MVAS CLI - Command Line Interface

Commands:
- mvas serve: Start the MVAS server
- mvas package: Create a .mvapp package (v1 script or v2 native plugin)
- mvas validate: Validate a .mvapp package
- mvas inspect: Run inspection on image(s)
- mvas init: Initialize a new MVAS project
- mvas init-plugin: Initialize a new native plugin project
- mvas benchmark: Benchmark an application's performance
"""

import argparse
import json
import logging
import os
import platform
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def cmd_serve(args):
    """Start the MVAS server"""
    from mvas_runtime.server import run_server
    
    logger.info(f"Starting MVAS server on {args.host}:{args.port}")
    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


def cmd_package(args):
    """Create a .mvapp package (supports both v1 script and v2 native plugin)"""
    source_dir = Path(args.source)
    output_path = Path(args.output) if args.output else source_dir.with_suffix(".mvapp")
    
    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        sys.exit(1)
    
    # Validate required files
    manifest_path = source_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error("manifest.json not found in source directory")
        sys.exit(1)
    
    # Load and validate manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Detect package type
    package_type = manifest.get("package_type", "script")
    is_native = package_type == "native" or "plugin" in manifest
    
    if is_native:
        logger.info("Detected native plugin package (v2)")
        _validate_native_plugin(source_dir, manifest)
    else:
        logger.info("Detected script-based package (v1)")
        _validate_script_package(source_dir, manifest)
    
    # Create ZIP package
    logger.info(f"Creating package: {output_path}")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                # Skip build artifacts and temp files
                if any(skip in str(file_path) for skip in ['.git', '__pycache__', '.pyc', 'build/', '.obj']):
                    continue
                arcname = file_path.relative_to(source_dir)
                zf.write(file_path, arcname)
                logger.debug(f"  Added: {arcname}")
    
    # Get package info
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Package created: {output_path} ({size_mb:.2f} MB)")
    logger.info(f"Package type: {'native plugin' if is_native else 'script'}")
    
    return output_path


def _validate_script_package(source_dir: Path, manifest: dict):
    """Validate v1 script-based package"""
    # Check model file exists
    model_path = source_dir / manifest.get("model", {}).get("path", "model/model.onnx")
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)


def _validate_native_plugin(source_dir: Path, manifest: dict):
    """Validate v2 native plugin package"""
    plugin_config = manifest.get("plugin", {})
    binaries = plugin_config.get("binaries", {})
    
    if not binaries:
        logger.error("No plugin binaries specified in manifest")
        sys.exit(1)
    
    # Check at least one binary exists
    found_binary = False
    for platform_key, binary_path in binaries.items():
        if binary_path:
            full_path = source_dir / binary_path
            if full_path.exists():
                found_binary = True
                logger.info(f"  Found binary for {platform_key}: {binary_path}")
            else:
                logger.warning(f"  Binary not found for {platform_key}: {binary_path}")
    
    if not found_binary:
        logger.error("No valid plugin binaries found")
        sys.exit(1)
    
    # Check model files
    model_config = plugin_config.get("model", {})
    model_path = model_config.get("path", "models/model.engine")
    fallback_path = model_config.get("fallback", "models/model.onnx")
    
    if not (source_dir / model_path).exists():
        if fallback_path and (source_dir / fallback_path).exists():
            logger.warning(f"Primary model not found, fallback exists: {fallback_path}")
        else:
            logger.error(f"No model files found: {model_path} or {fallback_path}")
            sys.exit(1)


def cmd_validate(args):
    """Validate a .mvapp package"""
    app_path = Path(args.app_path)
    
    if not app_path.exists():
        logger.error(f"Package not found: {app_path}")
        sys.exit(1)
    
    logger.info(f"Validating package: {app_path}")
    
    errors = []
    warnings = []
    
    try:
        with zipfile.ZipFile(app_path, 'r') as zf:
            # Check for manifest
            if "manifest.json" not in zf.namelist():
                errors.append("manifest.json not found in package")
            else:
                # Parse manifest
                with zf.open("manifest.json") as f:
                    manifest = json.load(f)
                
                # Check required fields
                required_fields = ["mvas_version", "app", "model", "input", "output"]
                for field in required_fields:
                    if field not in manifest:
                        errors.append(f"Missing required field: {field}")
                
                # Check app metadata
                app_meta = manifest.get("app", {})
                for field in ["id", "name", "version"]:
                    if field not in app_meta:
                        errors.append(f"Missing app.{field}")
                
                # Check model file exists
                model_config = manifest.get("model", {})
                model_path = model_config.get("path", "")
                if model_path and model_path not in zf.namelist():
                    errors.append(f"Model file not found: {model_path}")
                
                # Check preprocessing config
                input_config = manifest.get("input", {})
                preproc_path = input_config.get("preprocessing", "")
                if preproc_path and preproc_path not in zf.namelist():
                    warnings.append(f"Preprocessing config not found: {preproc_path}")
                
                # Check postprocessing config
                output_config = manifest.get("output", {})
                postproc_path = output_config.get("postprocessing", "")
                if postproc_path and postproc_path not in zf.namelist():
                    warnings.append(f"Postprocessing config not found: {postproc_path}")
    
    except zipfile.BadZipFile:
        errors.append("Invalid ZIP file")
    except json.JSONDecodeError:
        errors.append("Invalid JSON in manifest.json")
    except Exception as e:
        errors.append(f"Validation error: {e}")
    
    # Report results
    if warnings:
        for warning in warnings:
            logger.warning(f"⚠️  {warning}")
    
    if errors:
        for error in errors:
            logger.error(f"❌ {error}")
        logger.error("Validation FAILED")
        sys.exit(1)
    else:
        logger.info("✅ Validation PASSED")


def cmd_inspect(args):
    """Run inspection on image(s)"""
    import cv2
    
    # Import runtime components
    from mvas_runtime.app_manager import get_app_manager
    
    manager = get_app_manager()
    
    # Load application
    logger.info(f"Loading application: {args.app}")
    app = manager.load_app(args.app)
    
    # Process images
    for image_path in args.images:
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            continue
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Failed to read image: {image_path}")
            continue
        
        # Run inspection
        result = manager.inspect(app_id=app.app_id, image=image)
        
        # Print result
        print(f"\n{image_path.name}:")
        print(f"  Decision: {result.decision.value.upper()}")
        print(f"  Anomaly Score: {result.anomaly_score:.4f}")
        print(f"  Confidence: {result.confidence:.4f}")
        print(f"  Inference Time: {result.inference_time_ms:.2f} ms")
        
        # Save visualization if requested
        if args.output:
            from mvas_runtime.outputs.visualization import Visualizer
            
            viz = Visualizer()
            vis_image = viz.create_result_visualization(
                image=image,
                decision=result.decision.value,
                anomaly_score=result.anomaly_score,
            )
            
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{image_path.stem}_result.jpg"
            cv2.imwrite(str(output_path), vis_image)
            print(f"  Saved: {output_path}")


def cmd_init(args):
    """Initialize a new MVAS application project"""
    project_dir = Path(args.name)
    
    if project_dir.exists():
        logger.error(f"Directory already exists: {project_dir}")
        sys.exit(1)
    
    logger.info(f"Creating new MVAS project: {project_dir}")
    
    # Create directory structure
    (project_dir / "model").mkdir(parents=True)
    (project_dir / "preprocessing").mkdir()
    (project_dir / "postprocessing").mkdir()
    (project_dir / "assets").mkdir()
    (project_dir / "ui").mkdir()
    
    # Create manifest.json
    manifest = {
        "mvas_version": "1.0.0",
        "app": {
            "id": args.name.lower().replace(" ", "-"),
            "name": args.name,
            "version": "1.0.0",
            "description": "MVAS application",
            "author": "",
            "tags": []
        },
        "model": {
            "type": "anomaly_detection",
            "algorithm": "patchcore",
            "framework": "onnx",
            "path": "model/model.onnx",
            "runtime": {
                "device": "auto",
                "precision": "fp32"
            }
        },
        "input": {
            "type": "image",
            "color_mode": "RGB",
            "resolution": {"width": 256, "height": 256},
            "preprocessing": "preprocessing/transforms.json"
        },
        "output": {
            "type": "anomaly_map",
            "postprocessing": "postprocessing/rules.json",
            "save_images": True
        }
    }
    
    with open(project_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create transforms.json
    transforms = {
        "pipeline": [
            {"op": "resize", "params": {"width": 256, "height": 256}},
            {"op": "bgr2rgb", "params": {}},
            {"op": "normalize", "params": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }},
            {"op": "to_tensor", "params": {"dtype": "float32"}},
            {"op": "add_batch_dim", "params": {}}
        ]
    }
    
    with open(project_dir / "preprocessing" / "transforms.json", 'w') as f:
        json.dump(transforms, f, indent=2)
    
    # Create rules.json
    rules = {
        "thresholds": {
            "anomaly_score": {
                "pass": 0.3,
                "review": 0.7
            }
        },
        "outputs": {
            "include_anomaly_map": True,
            "include_heatmap_overlay": True
        }
    }
    
    with open(project_dir / "postprocessing" / "rules.json", 'w') as f:
        json.dump(rules, f, indent=2)
    
    logger.info(f"Project created: {project_dir}")
    logger.info("Next steps:")
    logger.info(f"  1. Add your model file to {project_dir}/model/model.onnx")
    logger.info(f"  2. Edit manifest.json to configure your application")
    logger.info(f"  3. Package with: mvas package {project_dir}")


def cmd_init_plugin(args):
    """Initialize a new native plugin project (v2)"""
    project_dir = Path(args.name)
    
    if project_dir.exists():
        logger.error(f"Directory already exists: {project_dir}")
        sys.exit(1)
    
    logger.info(f"Creating new native plugin project: {project_dir}")
    
    # Create directory structure
    (project_dir / "src").mkdir(parents=True)
    (project_dir / "include").mkdir()
    (project_dir / "models").mkdir()
    (project_dir / "bin" / "windows").mkdir(parents=True)
    (project_dir / "bin" / "linux").mkdir(parents=True)
    (project_dir / "bin" / "linux-arm64").mkdir(parents=True)
    (project_dir / "config").mkdir()
    (project_dir / "assets").mkdir()
    (project_dir / "test" / "test_images").mkdir(parents=True)
    
    # Create manifest.json
    manifest = {
        "mvas_version": "2.0.0",
        "manifest_version": "2.0.0",
        "package_type": "native",
        
        "app": {
            "id": args.name.lower().replace(" ", "-").replace("_", "-"),
            "name": args.name,
            "version": "1.0.0",
            "description": "Native MVAS plugin application",
            "author": "",
            "tags": []
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
            "dependencies": {}
        },
        
        "interface": {
            "input": {
                "type": "image",
                "color_mode": "RGB",
                "resolution": {"width": 256, "height": 256}
            },
            "output": {
                "type": "anomaly_map"
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
                }
            ]
        },
        
        "requirements": {
            "min_mvas_version": "2.0.0",
            "gpu_required": True,
            "cuda_version": ">=11.8"
        }
    }
    
    with open(project_dir / "manifest.json", 'w') as f:
        json.dump(manifest, f, indent=2)
    
    # Create CMakeLists.txt
    cmake_content = '''cmake_minimum_required(VERSION 3.18)
project({name} VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(OpenCV REQUIRED COMPONENTS core imgproc)

# Plugin library
add_library(mvas_plugin SHARED
    src/plugin.cpp
)

target_include_directories(mvas_plugin PRIVATE
    ${{CMAKE_CURRENT_SOURCE_DIR}}/include
    ${{OpenCV_INCLUDE_DIRS}}
)

target_link_libraries(mvas_plugin PRIVATE
    ${{OpenCV_LIBS}}
)

# Platform-specific settings
if(WIN32)
    set_target_properties(mvas_plugin PROPERTIES
        OUTPUT_NAME "plugin"
        SUFFIX ".dll"
    )
else()
    set_target_properties(mvas_plugin PROPERTIES
        OUTPUT_NAME "plugin"
        SUFFIX ".so"
    )
endif()
'''.format(name=args.name.lower().replace(" ", "_"))
    
    with open(project_dir / "CMakeLists.txt", 'w') as f:
        f.write(cmake_content)
    
    # Create plugin.cpp template
    plugin_cpp = '''// {name} - MVAS Native Plugin
// Implement the standard MVAS plugin interface

#include "mvas_plugin.h"
#include <string>
#include <chrono>

// Plugin state
static struct {{
    bool initialized = false;
    float threshold = 0.5f;
    std::string last_error;
    MVASPluginInfo info;
}} g_plugin;

// ============================================================================
// Required Functions
// ============================================================================

MVAS_EXPORT const MVASPluginInfo* mvas_get_info(void) {{
    g_plugin.info.name = "{name}";
    g_plugin.info.version = "1.0.0";
    g_plugin.info.author = "Your Name";
    g_plugin.info.description = "Native MVAS plugin";
    g_plugin.info.model_type = "anomaly_detection";
    g_plugin.info.input_width = 256;
    g_plugin.info.input_height = 256;
    g_plugin.info.input_format = "RGB";
    return &g_plugin.info;
}}

MVAS_EXPORT int32_t mvas_init(const char* model_path, const char* config_json) {{
    // TODO: Load your model here
    // Example: Load ONNX, TensorRT engine, etc.
    
    g_plugin.initialized = true;
    return 0;
}}

MVAS_EXPORT int32_t mvas_infer(const MVASImage* image, MVASResult* result) {{
    if (!g_plugin.initialized) {{
        g_plugin.last_error = "Plugin not initialized";
        return -1;
    }}
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // TODO: Implement your inference logic here
    // 1. Preprocess image
    // 2. Run model inference
    // 3. Postprocess results
    
    // Example: Return dummy result
    result->decision = "pass";
    result->confidence = 0.95f;
    result->anomaly_score = 0.05f;
    
    auto end = std::chrono::high_resolution_clock::now();
    result->inference_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    
    return 0;
}}

MVAS_EXPORT void mvas_cleanup(void) {{
    // TODO: Release resources
    g_plugin.initialized = false;
}}

MVAS_EXPORT const char* mvas_get_error(void) {{
    return g_plugin.last_error.c_str();
}}

// ============================================================================
// Optional Functions
// ============================================================================

MVAS_EXPORT int32_t mvas_warmup(int32_t iterations) {{
    // TODO: Warmup the model
    return 0;
}}

MVAS_EXPORT void mvas_free_result(MVASResult* result) {{
    // Free any allocated memory in result
}}
'''.format(name=args.name)
    
    with open(project_dir / "src" / "plugin.cpp", 'w') as f:
        f.write(plugin_cpp)
    
    # Create mvas_plugin.h header
    _create_plugin_header(project_dir / "include")
    
    # Create default config
    default_config = {"threshold": 0.5}
    with open(project_dir / "config" / "default_params.json", 'w') as f:
        json.dump(default_config, f, indent=2)
    
    # Create model config placeholder
    model_config = {"description": "Add your model configuration here"}
    with open(project_dir / "models" / "model_config.json", 'w') as f:
        json.dump(model_config, f, indent=2)
    
    # Create build script
    build_sh = '''#!/bin/bash
# Build script for Linux

set -e
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Copy to bin directory
cp plugin.so ../bin/linux/
echo "Build complete!"
'''
    with open(project_dir / "build.sh", 'w') as f:
        f.write(build_sh)
    
    build_ps1 = '''# Build script for Windows
$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path build | Out-Null
Push-Location build

cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

# Copy to bin directory
Copy-Item Release\\plugin.dll ..\\bin\\windows\\
Pop-Location

Write-Host "Build complete!"
'''
    with open(project_dir / "build.ps1", 'w') as f:
        f.write(build_ps1)
    
    logger.info(f"Native plugin project created: {project_dir}")
    logger.info("Next steps:")
    logger.info(f"  1. Implement your inference logic in {project_dir}/src/plugin.cpp")
    logger.info(f"  2. Add your model files to {project_dir}/models/")
    logger.info(f"  3. Build with: cd {project_dir} && ./build.sh (or build.ps1 on Windows)")
    logger.info(f"  4. Package with: mvas package {project_dir}")


def _create_plugin_header(include_dir: Path):
    """Create the mvas_plugin.h header file"""
    header = '''// mvas_plugin.h - MVAS Native Plugin Interface
// Version: 2.0.0
// 
// All native plugins must implement the required functions defined here.
// Compile your plugin as a shared library (DLL on Windows, SO on Linux).

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
    float inference_time_ms;        // Inference time in milliseconds
    
    // Optional: Anomaly heatmap (set to NULL if not available)
    float* anomaly_map;             // width*height float array
    int32_t anomaly_map_width;
    int32_t anomaly_map_height;
    
    // Optional: Bounding boxes [x1, y1, x2, y2, score, class]
    float* bboxes;                  // NULL or array of boxes
    int32_t num_bboxes;
    
    // Optional: Visualization image
    uint8_t* visualization;         // RGB image data
    int32_t viz_width;
    int32_t viz_height;
    
    // Optional: Custom JSON details
    const char* details_json;       // NULL or JSON string
} MVASResult;

// Plugin information structure
typedef struct {
    const char* name;               // "My Inspection App"
    const char* version;            // "1.0.0"
    const char* author;             // "Company Name"
    const char* description;        // "Detects defects..."
    const char* model_type;         // "anomaly_detection", "classification", etc.
    int32_t input_width;            // Expected input width
    int32_t input_height;           // Expected input height
    const char* input_format;       // "RGB" or "BGR"
} MVASPluginInfo;

// Configuration parameter for runtime updates
typedef struct {
    const char* key;                // "threshold"
    const char* value;              // "0.5"
} MVASConfigParam;

// ============================================================================
// Required Functions - Must be implemented by all plugins
// ============================================================================

// Get plugin information (called before init)
MVAS_EXPORT const MVASPluginInfo* mvas_get_info(void);

// Initialize plugin with model path and configuration
// Returns: 0 on success, non-zero on error
MVAS_EXPORT int32_t mvas_init(const char* model_path, const char* config_json);

// Run inference on a single image
// Returns: 0 on success, non-zero on error
MVAS_EXPORT int32_t mvas_infer(const MVASImage* image, MVASResult* result);

// Cleanup and release all resources
MVAS_EXPORT void mvas_cleanup(void);

// Get last error message
MVAS_EXPORT const char* mvas_get_error(void);

// ============================================================================
// Optional Functions - Implement for enhanced functionality
// ============================================================================

// Warmup the model (run dummy inferences to initialize GPU)
MVAS_EXPORT int32_t mvas_warmup(int32_t iterations);

// Update configuration at runtime
MVAS_EXPORT int32_t mvas_set_config(const MVASConfigParam* params, int32_t num_params);

// Free memory allocated in MVASResult by the plugin
MVAS_EXPORT void mvas_free_result(MVASResult* result);

#ifdef __cplusplus
}
#endif

#endif // MVAS_PLUGIN_H
'''
    with open(include_dir / "mvas_plugin.h", 'w') as f:
        f.write(header)


def cmd_benchmark(args):
    """Benchmark an application's performance"""
    import cv2
    
    from mvas_runtime.app_manager import get_app_manager
    
    manager = get_app_manager()
    
    # Load application
    logger.info(f"Loading application: {args.app}")
    app = manager.load_app(args.app)
    
    # Create or load test image
    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            logger.error(f"Failed to load image: {args.image}")
            sys.exit(1)
    else:
        # Create dummy image
        info = app.info
        if hasattr(info, 'input_width'):
            w, h = info.input_width, info.input_height
        else:
            w, h = info.input_resolution
        image = (np.random.rand(h, w, 3) * 255).astype(np.uint8)
        logger.info(f"Using random {w}x{h} image for benchmark")
    
    iterations = args.iterations
    warmup = args.warmup
    
    # Warmup
    logger.info(f"Warming up with {warmup} iterations...")
    for _ in range(warmup):
        manager.inspect(app_id=app.app_id, image=image)
    
    # Benchmark
    logger.info(f"Running {iterations} iterations...")
    times = []
    
    import numpy as np
    
    for i in range(iterations):
        result = manager.inspect(app_id=app.app_id, image=image)
        times.append(result.inference_time_ms)
        
        if (i + 1) % 100 == 0:
            logger.info(f"  Completed {i + 1}/{iterations}")
    
    times = np.array(times)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"BENCHMARK RESULTS: {app.info.name}")
    print("=" * 60)
    print(f"  Iterations:     {iterations}")
    print(f"  Avg inference:  {np.mean(times):.2f} ms")
    print(f"  Std deviation:  {np.std(times):.2f} ms")
    print(f"  Min:            {np.min(times):.2f} ms")
    print(f"  Max:            {np.max(times):.2f} ms")
    print(f"  P50 (median):   {np.percentile(times, 50):.2f} ms")
    print(f"  P95:            {np.percentile(times, 95):.2f} ms")
    print(f"  P99:            {np.percentile(times, 99):.2f} ms")
    print(f"  Throughput:     {1000.0 / np.mean(times):.1f} FPS")
    print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        prog="mvas",
        description="MVAS - Machine Vision Application Standard CLI (v2 with Native Plugin support)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the MVAS server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # package command
    package_parser = subparsers.add_parser("package", help="Create a .mvapp package (v1 or v2)")
    package_parser.add_argument("source", help="Source directory")
    package_parser.add_argument("-o", "--output", help="Output .mvapp file path")
    
    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a .mvapp package")
    validate_parser.add_argument("app_path", help="Path to .mvapp file")
    
    # inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Run inspection on image(s)")
    inspect_parser.add_argument("app", help="Path to .mvapp file")
    inspect_parser.add_argument("images", nargs="+", help="Image file(s) to inspect")
    inspect_parser.add_argument("-o", "--output", help="Output directory for visualizations")
    
    # init command (v1 script)
    init_parser = subparsers.add_parser("init", help="Initialize a new script-based project (v1)")
    init_parser.add_argument("name", help="Project name")
    
    # init-plugin command (v2 native)
    init_plugin_parser = subparsers.add_parser("init-plugin", help="Initialize a new native plugin project (v2)")
    init_plugin_parser.add_argument("name", help="Project name")
    
    # benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Benchmark an application")
    benchmark_parser.add_argument("app", help="Path to .mvapp file")
    benchmark_parser.add_argument("-i", "--image", help="Test image (optional, uses random if not provided)")
    benchmark_parser.add_argument("-n", "--iterations", type=int, default=100, help="Number of iterations")
    benchmark_parser.add_argument("-w", "--warmup", type=int, default=10, help="Warmup iterations")
    
    args = parser.parse_args()
    
    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "package":
        cmd_package(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "init":
        cmd_init(args)
    elif args.command == "init-plugin":
        cmd_init_plugin(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

