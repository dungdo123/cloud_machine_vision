"""
MVAS Application Loader

Handles loading, validation, and management of .mvapp packages.
Supports both v1 (script-based) and v2 (native plugin) packages.
"""

import json
import logging
import platform
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime

from .models import (
    Manifest, AppMetadata, ModelConfig, InputConfig, OutputConfig,
    TransformPipeline, RulesConfig, AppInfo, PackageType, PluginAppInfo,
    PluginState
)
from .inference_engine import InferenceEngine
from .preprocessing import PreprocessingPipeline
from .plugin_loader import NativePlugin, NativePluginInfo, PluginManager

logger = logging.getLogger(__name__)


class AppLoadError(Exception):
    """Exception raised when app loading fails"""
    pass


class AppInstance:
    """
    Represents a loaded application instance (v1 script-based).
    
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
        self.package_type = PackageType.SCRIPT
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
            model_type=self.manifest.model.type.value if self.manifest.model else "unknown",
            algorithm=self.manifest.model.algorithm if self.manifest.model else "",
            input_resolution=(
                self.manifest.input.resolution["width"] if self.manifest.input else 256,
                self.manifest.input.resolution["height"] if self.manifest.input else 256
            ),
            loaded_at=self.loaded_at,
            package_type=PackageType.SCRIPT,
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


class NativePluginInstance:
    """
    Represents a loaded native plugin application (v2).
    
    Contains:
    - Manifest configuration
    - Loaded native plugin (DLL/SO)
    - Plugin information
    """
    
    def __init__(
        self,
        app_id: str,
        manifest: Manifest,
        app_dir: Path,
        plugin: NativePlugin,
    ):
        self.app_id = app_id
        self.manifest = manifest
        self.app_dir = app_dir
        self.plugin = plugin
        self.loaded_at = datetime.now()
        self.package_type = PackageType.NATIVE
        self._plugin_info = plugin.get_info()
    
    @property
    def info(self) -> PluginAppInfo:
        """Get plugin app info for API responses"""
        stats = self.plugin.get_stats()
        return PluginAppInfo(
            id=self.app_id,
            name=self._plugin_info.name or self.manifest.app.name,
            version=self._plugin_info.version or self.manifest.app.version,
            description=self._plugin_info.description or self.manifest.app.description,
            author=self._plugin_info.author or self.manifest.app.author,
            model_type=self._plugin_info.model_type,
            input_width=self._plugin_info.input_width,
            input_height=self._plugin_info.input_height,
            input_format=self._plugin_info.input_format,
            package_type=PackageType.NATIVE,
            state=PluginState.READY if self.plugin.is_loaded else PluginState.ERROR,
            load_time_ms=stats["load_time_ms"],
            avg_inference_ms=stats["avg_inference_ms"],
            total_inferences=stats["total_inferences"],
            loaded_at=self.loaded_at,
        )
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get plugin statistics"""
        return self.plugin.get_stats()
    
    def update_stats(self, decision: str, inference_time_ms: float):
        """Stats are updated internally by the plugin"""
        pass  # Plugin tracks its own stats
    
    def cleanup(self):
        """Clean up resources when unloading"""
        try:
            if self.plugin:
                self.plugin.cleanup()
            
            # Remove extracted directory
            if self.app_dir.exists() and "plugins" in str(self.app_dir):
                shutil.rmtree(self.app_dir, ignore_errors=True)
                
        except Exception as e:
            logger.warning(f"Cleanup error for plugin {self.app_id}: {e}")


class AppLoader:
    """
    Loads and validates .mvapp application packages.
    
    Supports both package types:
    - v1 (script): ONNX/Python based applications
    - v2 (native): DLL/SO compiled plugins
    
    Workflow:
    1. Extract ZIP package to temp directory
    2. Parse and validate manifest.json
    3. Detect package type (script vs native)
    4. Load appropriate engine (InferenceEngine or NativePlugin)
    5. Return AppInstance or NativePluginInstance
    """
    
    REQUIRED_FILES = ["manifest.json"]
    MANIFEST_SCHEMA_VERSION = "2.0.0"
    
    def __init__(self, temp_dir: Optional[Path] = None, plugins_dir: Optional[Path] = None):
        """
        Initialize the app loader.
        
        Args:
            temp_dir: Directory for extracting script packages
            plugins_dir: Directory for extracting plugin packages
        """
        self.temp_dir = temp_dir or Path(tempfile.gettempdir()) / "mvas_apps"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        self.plugins_dir = plugins_dir or Path("plugins")
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
    
    def load(self, app_path: str | Path) -> Union[AppInstance, NativePluginInstance]:
        """
        Load an application from a .mvapp package.
        
        Automatically detects package type (script or native) and loads
        the appropriate engine.
        
        Args:
            app_path: Path to the .mvapp file
            
        Returns:
            AppInstance (v1) or NativePluginInstance (v2)
            
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
            # Load manifest first to determine package type
            manifest = self._peek_manifest(app_path)
            
            # Route to appropriate loader based on package type
            if manifest.is_native_plugin():
                return self._load_native_plugin(app_path, manifest)
            else:
                return self._load_script_app(app_path, manifest)
            
        except AppLoadError:
            raise
        except Exception as e:
            logger.exception(f"Failed to load application: {e}")
            raise AppLoadError(f"Failed to load application: {e}")
    
    def _peek_manifest(self, app_path: Path) -> Manifest:
        """Read manifest without fully extracting the package"""
        try:
            with zipfile.ZipFile(app_path, 'r') as zf:
                if "manifest.json" not in zf.namelist():
                    raise AppLoadError("manifest.json not found in package")
                
                with zf.open("manifest.json") as f:
                    data = json.load(f)
                    logger.debug(f"Raw manifest data: {data}")
                    
                    # Debug: check plugin.binaries specifically
                    if 'plugin' in data and 'binaries' in data['plugin']:
                        logger.info(f"Raw binaries from manifest: {data['plugin']['binaries']}")
                    
                    manifest = Manifest(**data)
                    
                    # Debug: check parsed binaries
                    if manifest.plugin and manifest.plugin.binaries:
                        logger.info(f"Parsed windows_x64: {manifest.plugin.binaries.windows_x64}")
                    
                    return manifest
        except zipfile.BadZipFile:
            raise AppLoadError(f"Invalid ZIP file: {app_path}")
        except json.JSONDecodeError as e:
            raise AppLoadError(f"Invalid JSON in manifest.json: {e}")
    
    def _load_native_plugin(self, app_path: Path, manifest: Manifest) -> NativePluginInstance:
        """Load a native plugin package (v2)"""
        logger.info(f"Loading native plugin: {manifest.app.name} v{manifest.app.version}")
        
        # Extract to plugins directory with unique timestamp to avoid locked files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extract_dir = self.plugins_dir / f"{manifest.app.id}_{timestamp}"
        
        # Try to clean up old plugin directories (ignore errors for locked files)
        for old_dir in self.plugins_dir.glob(f"{manifest.app.id}_*"):
            try:
                if old_dir != extract_dir:
                    shutil.rmtree(old_dir, ignore_errors=True)
            except Exception:
                pass  # Ignore - files may be locked
        
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(app_path, 'r') as zf:
            zf.extractall(extract_dir)
        
        # Determine platform and get binary path
        platform_key = self._get_platform_key()
        
        if not manifest.plugin or not manifest.plugin.binaries:
            raise AppLoadError("Native plugin manifest missing 'plugin.binaries' configuration")
        
        # Get binary path for current platform
        binaries = manifest.plugin.binaries
        binary_path = None
        
        # Try using the helper method first, then fall back to direct access
        binary_path = binaries.get_binary_for_platform(platform_key)
        
        # Debug logging
        logger.info(f"Platform: {platform_key}")
        logger.info(f"Binaries object: {binaries}")
        logger.info(f"Binary path from manifest: {binary_path}")
        
        if not binary_path:
            raise AppLoadError(f"No binary available for platform: {platform_key}")
        
        plugin_path = extract_dir / binary_path
        logger.info(f"Full plugin path: {plugin_path}")
        
        if not plugin_path.exists():
            # Try to find the plugin.dll in the directory
            if plugin_path.is_dir():
                dll_path = plugin_path / "plugin.dll"
                so_path = plugin_path / "plugin.so"
                if dll_path.exists():
                    plugin_path = dll_path
                elif so_path.exists():
                    plugin_path = so_path
                else:
                    raise AppLoadError(f"Plugin binary not found in directory: {plugin_path}")
            else:
                raise AppLoadError(f"Plugin binary not found: {plugin_path}")
        
        # Get model path
        model_path = extract_dir / manifest.plugin.model.path
        if not model_path.exists():
            # Try fallback
            if manifest.plugin.model.fallback:
                model_path = extract_dir / manifest.plugin.model.fallback
        
        if not model_path.exists():
            raise AppLoadError(f"Model file not found: {model_path}")
        
        # Get dependency paths
        deps_paths = []
        if manifest.plugin.dependencies:
            deps = manifest.plugin.dependencies.get(platform_key, [])
            for dep in deps:
                dep_path = extract_dir / dep
                if dep_path.exists():
                    deps_paths.append(dep_path)
        
        # Build config JSON
        config_json = "{}"
        if manifest.configuration and manifest.configuration.default_config:
            config_path = extract_dir / manifest.configuration.default_config
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_json = f.read()
        
        # Load the native plugin
        plugin = NativePlugin(
            plugin_path=plugin_path,
            model_path=model_path,
            config_json=config_json,
            deps_paths=deps_paths
        )
        
        # Create instance
        instance = NativePluginInstance(
            app_id=manifest.app.id,
            manifest=manifest,
            app_dir=extract_dir,
            plugin=plugin
        )
        
        logger.info(f"Successfully loaded native plugin: {manifest.app.id}")
        return instance
    
    def _load_script_app(self, app_path: Path, manifest: Manifest) -> AppInstance:
        """Load a script-based application (v1)"""
        logger.info(f"Loading script app: {manifest.app.name} v{manifest.app.version}")
        
        # Extract package
        extract_dir = self._extract_package(app_path)
        logger.debug(f"Extracted to: {extract_dir}")
        
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
    
    def _get_platform_key(self) -> str:
        """Get the platform key for binary selection"""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if system == "windows":
            return "windows-x64"
        elif system == "linux":
            if "arm" in machine or "aarch64" in machine:
                return "linux-arm64"
            return "linux-x64"
        elif system == "darwin":
            if "arm" in machine:
                return "macos-arm64"
            return "macos-x64"
        return "linux-x64"
    
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

