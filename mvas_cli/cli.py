"""
MVAS CLI - Command Line Interface

Commands:
- mvas serve: Start the MVAS server
- mvas package: Create a .mvapp package
- mvas validate: Validate a .mvapp package
- mvas inspect: Run inspection on image(s)
- mvas list: List loaded applications
"""

import argparse
import json
import logging
import os
import shutil
import sys
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
    """Create a .mvapp package"""
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
    
    # Check model file exists
    model_path = source_dir / manifest.get("model", {}).get("path", "model/model.onnx")
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        sys.exit(1)
    
    # Create ZIP package
    logger.info(f"Creating package: {output_path}")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for file_path in source_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(source_dir)
                zf.write(file_path, arcname)
                logger.debug(f"  Added: {arcname}")
    
    # Get package info
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Package created: {output_path} ({size_mb:.2f} MB)")
    
    return output_path


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


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        prog="mvas",
        description="MVAS - Machine Vision Application Standard CLI"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # serve command
    serve_parser = subparsers.add_parser("serve", help="Start the MVAS server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # package command
    package_parser = subparsers.add_parser("package", help="Create a .mvapp package")
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
    
    # init command
    init_parser = subparsers.add_parser("init", help="Initialize a new MVAS project")
    init_parser.add_argument("name", help="Project name")
    
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

