"""
MVAS Application Loader

Handles loading, validation, and management of .mvapp packages.
"""

import json
import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

from .models import (
    Manifest, AppMetadata, ModelConfig, InputConfig, OutputConfig,
    TransformPipeline, RulesConfig, AppInfo
)
from .inference_engine import InferenceEngine
from .preprocessing import PreprocessingPipeline

logger = logging.getLogger(__name__)


class AppLoadError(Exception):
    """Exception raised when app loading fails"""
    pass


class AppInstance:
    """
    Represents a loaded application instance.
    
    Contains all resources needed to run inference:
    - Manifest configuration
    - Loaded inference engine
    - Preprocessing pipeline
    - Postprocessing rules
    """
    
    def __init__(
        self,
        app_id: str,
        manifest: Manifest,
        app_dir: Path,
        inference_engine: InferenceEngine,
        preprocessing: PreprocessingPipeline,
        rules: RulesConfig,
    ):
        self.app_id = app_id
        self.manifest = manifest
        self.app_dir = app_dir
        self.inference_engine = inference_engine
        self.preprocessing = preprocessing
        self.rules = rules
        self.loaded_at = datetime.now()
        self._stats = {
            "total_inferences": 0,
            "pass_count": 0,
            "fail_count": 0,
            "review_count": 0,
            "error_count": 0,
            "total_inference_time_ms": 0.0,
        }
    
    @property
    def info(self) -> AppInfo:
        """Get app info for API responses"""
        return AppInfo(
            id=self.app_id,
            name=self.manifest.app.name,
            version=self.manifest.app.version,
            description=self.manifest.app.description,
            model_type=self.manifest.model.type.value,
            algorithm=self.manifest.model.algorithm,
            input_resolution=(
                self.manifest.input.resolution["width"],
                self.manifest.input.resolution["height"]
            ),
            loaded_at=self.loaded_at,
        )
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get inference statistics"""
        stats = self._stats.copy()
        if stats["total_inferences"] > 0:
            stats["avg_inference_time_ms"] = (
                stats["total_inference_time_ms"] / stats["total_inferences"]
            )
        else:
            stats["avg_inference_time_ms"] = 0.0
        return stats
    
    def update_stats(self, decision: str, inference_time_ms: float):
        """Update statistics after inference"""
        self._stats["total_inferences"] += 1
        self._stats["total_inference_time_ms"] += inference_time_ms
        
        if decision == "pass":
            self._stats["pass_count"] += 1
        elif decision == "fail":
            self._stats["fail_count"] += 1
        elif decision == "review":
            self._stats["review_count"] += 1
        else:
            self._stats["error_count"] += 1
    
    def cleanup(self):
        """Clean up resources when unloading"""
        try:
            # Cleanup inference engine
            if self.inference_engine:
                self.inference_engine.cleanup()
            
            # Remove temp directory if exists
            if self.app_dir.exists() and "temp" in str(self.app_dir):
                shutil.rmtree(self.app_dir, ignore_errors=True)
                
        except Exception as e:
            logger.warning(f"Cleanup error for app {self.app_id}: {e}")


class AppLoader:
    """
    Loads and validates .mvapp application packages.
    
    Workflow:
    1. Extract ZIP package to temp directory
    2. Parse and validate manifest.json
    3. Load model into inference engine
    4. Setup preprocessing pipeline
    5. Load postprocessing rules
    6. Return AppInstance
    """
    
    REQUIRED_FILES = ["manifest.json"]
    MANIFEST_SCHEMA_VERSION = "1.0.0"
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize the app loader.
        
        Args:
            temp_dir: Directory for extracting packages. If None, uses system temp.
        """
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "mvas_apps"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self, app_path: str | Path) -> AppInstance:
        """
        Load an application from a .mvapp package.
        
        Args:
            app_path: Path to the .mvapp file
            
        Returns:
            AppInstance ready for inference
            
        Raises:
            AppLoadError: If loading fails
        """
        app_path = Path(app_path)
        
        if not app_path.exists():
            raise AppLoadError(f"Application file not found: {app_path}")
        
        if not app_path.suffix == ".mvapp":
            raise AppLoadError(f"Invalid file extension. Expected .mvapp, got {app_path.suffix}")
        
        logger.info(f"Loading application from: {app_path}")
        
        try:
            # Extract package
            extract_dir = self._extract_package(app_path)
            logger.debug(f"Extracted to: {extract_dir}")
            
            # Load and validate manifest
            manifest = self._load_manifest(extract_dir)
            logger.info(f"Loaded manifest for app: {manifest.app.name} v{manifest.app.version}")
            
            # Validate all required files exist
            self._validate_package_contents(extract_dir, manifest)
            
            # Load inference engine
            inference_engine = self._load_inference_engine(extract_dir, manifest)
            
            # Load preprocessing pipeline
            preprocessing = self._load_preprocessing(extract_dir, manifest)
            
            # Load postprocessing rules
            rules = self._load_rules(extract_dir, manifest)
            
            # Create app instance
            app_instance = AppInstance(
                app_id=manifest.app.id,
                manifest=manifest,
                app_dir=extract_dir,
                inference_engine=inference_engine,
                preprocessing=preprocessing,
                rules=rules,
            )
            
            logger.info(f"Successfully loaded app: {manifest.app.id}")
            return app_instance
            
        except AppLoadError:
            raise
        except Exception as e:
            logger.exception(f"Failed to load application: {e}")
            raise AppLoadError(f"Failed to load application: {e}")
    
    def _extract_package(self, app_path: Path) -> Path:
        """Extract .mvapp ZIP to temp directory"""
        extract_dir = self.temp_dir / app_path.stem / datetime.now().strftime("%Y%m%d_%H%M%S")
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with zipfile.ZipFile(app_path, 'r') as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise AppLoadError(f"Invalid ZIP file: {app_path}")
        
        return extract_dir
    
    def _load_manifest(self, extract_dir: Path) -> Manifest:
        """Load and parse manifest.json"""
        manifest_path = extract_dir / "manifest.json"
        
        if not manifest_path.exists():
            raise AppLoadError("manifest.json not found in package")
        
        try:
            with open(manifest_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise AppLoadError(f"Invalid JSON in manifest.json: {e}")
        
        try:
            manifest = Manifest(**data)
        except Exception as e:
            raise AppLoadError(f"Invalid manifest structure: {e}")
        
        return manifest
    
    def _validate_package_contents(self, extract_dir: Path, manifest: Manifest):
        """Validate all required files exist in the package"""
        # Check model file
        model_path = extract_dir / manifest.model.path
        if not model_path.exists():
            raise AppLoadError(f"Model file not found: {manifest.model.path}")
        
        # Check additional model files
        for name, path in manifest.model.additional_files.items():
            file_path = extract_dir / path
            if not file_path.exists():
                raise AppLoadError(f"Additional file '{name}' not found: {path}")
        
        # Check preprocessing config
        preproc_path = extract_dir / manifest.input.preprocessing
        if not preproc_path.exists():
            logger.warning(f"Preprocessing config not found: {manifest.input.preprocessing}")
        
        # Check postprocessing config
        postproc_path = extract_dir / manifest.output.postprocessing
        if not postproc_path.exists():
            logger.warning(f"Postprocessing config not found: {manifest.output.postprocessing}")
    
    def _load_inference_engine(
        self, 
        extract_dir: Path, 
        manifest: Manifest
    ) -> InferenceEngine:
        """Load the inference engine with the model"""
        model_path = extract_dir / manifest.model.path
        
        # Load additional files
        additional_files = {}
        for name, path in manifest.model.additional_files.items():
            additional_files[name] = extract_dir / path
        
        engine = InferenceEngine(
            model_path=model_path,
            framework=manifest.model.framework,
            model_type=manifest.model.type,
            config=manifest.model.runtime,
            additional_files=additional_files,
        )
        
        return engine
    
    def _load_preprocessing(
        self, 
        extract_dir: Path, 
        manifest: Manifest
    ) -> PreprocessingPipeline:
        """Load the preprocessing pipeline"""
        preproc_path = extract_dir / manifest.input.preprocessing
        
        if preproc_path.exists():
            with open(preproc_path, 'r') as f:
                config = json.load(f)
            pipeline = TransformPipeline(**config)
        else:
            # Default preprocessing
            pipeline = TransformPipeline(pipeline=[])
        
        # Load ROI mask if specified
        roi_mask = None
        if manifest.input.roi_mask:
            mask_path = extract_dir / manifest.input.roi_mask
            if mask_path.exists():
                import cv2
                roi_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        return PreprocessingPipeline(
            pipeline=pipeline,
            target_size=(
                manifest.input.resolution["width"],
                manifest.input.resolution["height"]
            ),
            color_mode=manifest.input.color_mode,
            roi_mask=roi_mask,
        )
    
    def _load_rules(self, extract_dir: Path, manifest: Manifest) -> RulesConfig:
        """Load postprocessing rules"""
        rules_path = extract_dir / manifest.output.postprocessing
        
        if rules_path.exists():
            with open(rules_path, 'r') as f:
                config = json.load(f)
            return RulesConfig(**config)
        else:
            # Default rules
            return RulesConfig(
                thresholds={"anomaly_score": {"pass": 0.3, "review": 0.7}},
                outputs={"include_anomaly_map": True},
            )
    
    def validate_manifest(self, manifest_data: Dict) -> tuple[bool, str]:
        """
        Validate a manifest dictionary.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            Manifest(**manifest_data)
            return True, ""
        except Exception as e:
            return False, str(e)

