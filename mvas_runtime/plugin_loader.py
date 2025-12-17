"""
MVAS Native Plugin Loader

Loads and manages native DLL/SO plugins using ctypes.
Provides a Python interface to C-compiled vision applications.
"""

import ctypes
import os
import platform
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# C Structure Mappings (must match mvas_plugin.h)
# ============================================================================

class MVASImage(ctypes.Structure):
    """C structure for image input"""
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("channels", ctypes.c_int32),
        ("stride", ctypes.c_int32),
        ("format", ctypes.c_char_p),
    ]


class MVASResult(ctypes.Structure):
    """C structure for inference result"""
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
    """C structure for plugin information"""
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


class MVASConfigParam(ctypes.Structure):
    """C structure for configuration parameter"""
    _fields_ = [
        ("key", ctypes.c_char_p),
        ("value", ctypes.c_char_p),
    ]


# ============================================================================
# Python Data Classes
# ============================================================================

class PluginState(str, Enum):
    """Plugin lifecycle states"""
    LOADING = "loading"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    UNLOADED = "unloaded"


@dataclass
class NativePluginInfo:
    """Plugin information (Python-friendly)"""
    name: str = ""
    version: str = ""
    author: str = ""
    description: str = ""
    model_type: str = ""
    input_width: int = 256
    input_height: int = 256
    input_format: str = "RGB"


@dataclass
class NativeInferenceResult:
    """Inference result (Python-friendly)"""
    decision: str = "unknown"
    confidence: float = 0.0
    anomaly_score: float = 0.0
    inference_time_ms: float = 0.0
    anomaly_map: Optional[np.ndarray] = None
    bboxes: Optional[np.ndarray] = None
    visualization: Optional[np.ndarray] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class PluginStats:
    """Plugin statistics"""
    load_time_ms: float = 0.0
    total_inferences: int = 0
    total_inference_time_ms: float = 0.0
    avg_inference_ms: float = 0.0
    min_inference_ms: float = float('inf')
    max_inference_ms: float = 0.0


# ============================================================================
# Native Plugin Wrapper
# ============================================================================

class NativePlugin:
    """
    Wrapper for MVAS native plugins (DLL/SO).
    
    Loads a compiled plugin library and provides a Python interface
    to the standard MVAS plugin functions.
    """
    
    def __init__(
        self,
        plugin_path: Path,
        model_path: Path,
        config_json: str = "{}",
        deps_paths: Optional[list] = None
    ):
        """
        Initialize and load a native plugin.
        
        Args:
            plugin_path: Path to the DLL/SO file
            model_path: Path to the model file(s)
            config_json: JSON configuration string
            deps_paths: Additional dependency library paths
        """
        self.plugin_path = Path(plugin_path)
        self.model_path = Path(model_path)
        self._lib = None
        self._loaded = False
        self._info: Optional[NativePluginInfo] = None
        self._stats = PluginStats()
        
        # Load dependencies first (if any)
        if deps_paths:
            self._load_dependencies(deps_paths)
        
        # Load and initialize plugin
        start_time = time.perf_counter()
        self._load_library()
        self._setup_function_signatures()
        self._initialize(config_json)
        self._stats.load_time_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(f"Plugin loaded in {self._stats.load_time_ms:.1f}ms: {self.plugin_path.name}")
    
    def _load_dependencies(self, deps_paths: list):
        """Load dependency libraries before main plugin"""
        if not deps_paths:
            return
            
        # Get the first dependency's directory and add it to search path
        first_dep = Path(deps_paths[0]).resolve() if deps_paths else None
        if first_dep and first_dep.parent.exists():
            dep_dir = str(first_dep.parent)
            
            if platform.system() == "Windows":
                # Add to DLL search path
                try:
                    if hasattr(os, 'add_dll_directory'):
                        os.add_dll_directory(dep_dir)
                        logger.info(f"Added DLL search directory: {dep_dir}")
                except OSError as e:
                    logger.warning(f"Could not add DLL directory: {e}")
                
                # Also add to PATH
                current_path = os.environ.get('PATH', '')
                if dep_dir not in current_path:
                    os.environ['PATH'] = dep_dir + os.pathsep + current_path
                    
            else:
                # Linux: update LD_LIBRARY_PATH
                current_ld = os.environ.get('LD_LIBRARY_PATH', '')
                if dep_dir not in current_ld:
                    os.environ['LD_LIBRARY_PATH'] = dep_dir + ':' + current_ld
        
        # For Windows, we don't need to explicitly load each DLL
        # Just adding the directory to the search path is enough
        # The OS will load them when needed by the main plugin
        if platform.system() != "Windows":
            # Load each dependency explicitly on Linux
            for dep_path in deps_paths:
                dep_path = Path(dep_path).resolve()
                if dep_path.exists():
                    try:
                        ctypes.CDLL(str(dep_path), mode=ctypes.RTLD_GLOBAL)
                        logger.debug(f"Loaded dependency: {dep_path}")
                    except Exception as e:
                        logger.warning(f"Failed to load dependency {dep_path}: {e}")
    
    def _load_library(self):
        """Load the native library"""
        if not self.plugin_path.exists():
            raise FileNotFoundError(f"Plugin not found: {self.plugin_path}")
        
        # Convert to absolute path
        abs_plugin_path = self.plugin_path.resolve()
        plugin_dir = abs_plugin_path.parent
        
        try:
            if platform.system() == "Windows":
                self._load_library_windows(abs_plugin_path, plugin_dir)
            else:
                self._load_library_unix(abs_plugin_path, plugin_dir)
                
            logger.info(f"Successfully loaded plugin: {abs_plugin_path}")
        except OSError as e:
            raise RuntimeError(f"Failed to load plugin library '{abs_plugin_path}': {e}")
    
    def _load_library_windows(self, abs_plugin_path: Path, plugin_dir: Path):
        """Load library on Windows with proper DLL search path"""
        plugin_dir_str = str(plugin_dir)
        
        # Check if PyTorch is already loaded in Python
        torch_already_loaded = 'torch' in sys.modules
        if torch_already_loaded:
            logger.info("PyTorch already loaded in Python - using system torch DLLs")
        
        # Save current directory
        original_cwd = os.getcwd()
        
        try:
            # Change to plugin directory
            os.chdir(plugin_dir_str)
            logger.info(f"Changed to plugin directory: {plugin_dir_str}")
            
            # Add to PATH
            current_path = os.environ.get('PATH', '')
            if plugin_dir_str not in current_path:
                os.environ['PATH'] = plugin_dir_str + os.pathsep + current_path
            
            # Use add_dll_directory (Python 3.8+)
            try:
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(plugin_dir_str)
            except OSError:
                pass
            
            # If torch is NOT already loaded, pre-load the packaged DLLs
            if not torch_already_loaded:
                core_dlls = ['c10.dll', 'torch_cpu.dll', 'torch.dll']
                for dll_name in core_dlls:
                    dll_path = plugin_dir / dll_name
                    if dll_path.exists():
                        try:
                            ctypes.CDLL(str(dll_path))
                            logger.debug(f"Pre-loaded: {dll_name}")
                        except Exception as e:
                            logger.debug(f"Could not pre-load {dll_name}: {e}")
            
            # Load the main plugin
            self._lib = ctypes.CDLL(str(abs_plugin_path))
            
        finally:
            # Restore original directory
            os.chdir(original_cwd)
    
    def _load_library_unix(self, abs_plugin_path: Path, plugin_dir: Path):
        """Load library on Linux/macOS"""
        plugin_dir_str = str(plugin_dir)
        
        # Update LD_LIBRARY_PATH
        current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if plugin_dir_str not in current_ld_path:
            os.environ['LD_LIBRARY_PATH'] = plugin_dir_str + ':' + current_ld_path
        
        self._lib = ctypes.CDLL(str(abs_plugin_path), mode=ctypes.RTLD_GLOBAL)
    
    def _setup_function_signatures(self):
        """Set up C function signatures for type checking"""
        # mvas_get_info() -> MVASPluginInfo*
        self._lib.mvas_get_info.argtypes = []
        self._lib.mvas_get_info.restype = ctypes.POINTER(MVASPluginInfo)
        
        # mvas_init(model_path, config_json) -> int32
        self._lib.mvas_init.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        self._lib.mvas_init.restype = ctypes.c_int32
        
        # mvas_infer(image, result) -> int32
        self._lib.mvas_infer.argtypes = [
            ctypes.POINTER(MVASImage),
            ctypes.POINTER(MVASResult)
        ]
        self._lib.mvas_infer.restype = ctypes.c_int32
        
        # mvas_cleanup() -> void
        self._lib.mvas_cleanup.argtypes = []
        self._lib.mvas_cleanup.restype = None
        
        # mvas_get_error() -> char*
        self._lib.mvas_get_error.argtypes = []
        self._lib.mvas_get_error.restype = ctypes.c_char_p
        
        # Optional functions
        self._has_warmup = False
        self._has_set_config = False
        self._has_free_result = False
        
        try:
            self._lib.mvas_warmup.argtypes = [ctypes.c_int32]
            self._lib.mvas_warmup.restype = ctypes.c_int32
            self._has_warmup = True
        except AttributeError:
            pass
        
        try:
            self._lib.mvas_set_config.argtypes = [
                ctypes.POINTER(MVASConfigParam),
                ctypes.c_int32
            ]
            self._lib.mvas_set_config.restype = ctypes.c_int32
            self._has_set_config = True
        except AttributeError:
            pass
        
        try:
            self._lib.mvas_free_result.argtypes = [ctypes.POINTER(MVASResult)]
            self._lib.mvas_free_result.restype = None
            self._has_free_result = True
        except AttributeError:
            pass
    
    def _initialize(self, config_json: str):
        """Initialize the plugin with model and config"""
        model_path_bytes = str(self.model_path).encode('utf-8')
        config_bytes = config_json.encode('utf-8')
        
        result = self._lib.mvas_init(model_path_bytes, config_bytes)
        
        if result != 0:
            error = self._get_error()
            raise RuntimeError(f"Plugin initialization failed: {error}")
        
        self._loaded = True
        
        # Cache plugin info
        self._info = self.get_info()
    
    def _get_error(self) -> str:
        """Get last error message from plugin"""
        error = self._lib.mvas_get_error()
        if error:
            return error.decode('utf-8')
        return "Unknown error"
    
    def get_info(self) -> NativePluginInfo:
        """Get plugin information"""
        if self._info is not None:
            return self._info
        
        info_ptr = self._lib.mvas_get_info()
        if not info_ptr:
            return NativePluginInfo()
        
        info = info_ptr.contents
        return NativePluginInfo(
            name=info.name.decode('utf-8') if info.name else "",
            version=info.version.decode('utf-8') if info.version else "",
            author=info.author.decode('utf-8') if info.author else "",
            description=info.description.decode('utf-8') if info.description else "",
            model_type=info.model_type.decode('utf-8') if info.model_type else "",
            input_width=info.input_width,
            input_height=info.input_height,
            input_format=info.input_format.decode('utf-8') if info.input_format else "RGB"
        )
    
    def infer(self, image: np.ndarray) -> NativeInferenceResult:
        """
        Run inference on an image.
        
        Args:
            image: Input image as HWC uint8 numpy array (RGB or BGR)
            
        Returns:
            NativeInferenceResult with decision and details
        """
        if not self._loaded:
            raise RuntimeError("Plugin not initialized")
        
        # Ensure image is contiguous and correct type
        if not image.flags['C_CONTIGUOUS']:
            image = np.ascontiguousarray(image)
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Get image dimensions
        if len(image.shape) == 2:
            h, w = image.shape
            c = 1
        else:
            h, w, c = image.shape
        
        # Create C image structure
        mvas_image = MVASImage()
        mvas_image.data = image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        mvas_image.width = w
        mvas_image.height = h
        mvas_image.channels = c
        mvas_image.stride = w * c
        mvas_image.format = b"RGB"
        
        # Create result structure
        result = MVASResult()
        
        # Call inference
        start_time = time.perf_counter()
        ret = self._lib.mvas_infer(ctypes.byref(mvas_image), ctypes.byref(result))
        call_time = (time.perf_counter() - start_time) * 1000
        
        if ret != 0:
            error = self._get_error()
            raise RuntimeError(f"Inference failed: {error}")
        
        # Extract results
        inference_result = NativeInferenceResult(
            decision=result.decision.decode('utf-8') if result.decision else "unknown",
            confidence=result.confidence,
            anomaly_score=result.anomaly_score,
            inference_time_ms=result.inference_time_ms if result.inference_time_ms > 0 else call_time
        )
        
        # Extract anomaly map if available
        if result.anomaly_map and result.anomaly_map_width > 0 and result.anomaly_map_height > 0:
            map_size = result.anomaly_map_width * result.anomaly_map_height
            # Copy data to Python-owned array
            inference_result.anomaly_map = np.ctypeslib.as_array(
                result.anomaly_map,
                shape=(result.anomaly_map_height, result.anomaly_map_width)
            ).copy()
        
        # Extract bounding boxes if available
        if result.bboxes and result.num_bboxes > 0:
            inference_result.bboxes = np.ctypeslib.as_array(
                result.bboxes,
                shape=(result.num_bboxes, 6)  # x1, y1, x2, y2, score, class
            ).copy()
        
        # Extract visualization if available
        if result.visualization and result.viz_width > 0 and result.viz_height > 0:
            inference_result.visualization = np.ctypeslib.as_array(
                result.visualization,
                shape=(result.viz_height, result.viz_width, 3)
            ).copy()
        
        # Parse details JSON if available
        if result.details_json:
            try:
                import json
                details_str = result.details_json.decode('utf-8') if isinstance(result.details_json, bytes) else result.details_json
                inference_result.details = json.loads(details_str)
            except (json.JSONDecodeError, UnicodeDecodeError):
                inference_result.details = {"raw": str(result.details_json)}
        
        # Free plugin-allocated memory
        if self._has_free_result:
            self._lib.mvas_free_result(ctypes.byref(result))
        
        # Update statistics
        self._update_stats(inference_result.inference_time_ms)
        
        return inference_result
    
    def _update_stats(self, inference_time_ms: float):
        """Update inference statistics"""
        self._stats.total_inferences += 1
        self._stats.total_inference_time_ms += inference_time_ms
        self._stats.avg_inference_ms = (
            self._stats.total_inference_time_ms / self._stats.total_inferences
        )
        self._stats.min_inference_ms = min(self._stats.min_inference_ms, inference_time_ms)
        self._stats.max_inference_ms = max(self._stats.max_inference_ms, inference_time_ms)
    
    def set_config(self, params: Dict[str, Any]) -> bool:
        """
        Update plugin configuration at runtime.
        
        Args:
            params: Dictionary of key-value configuration parameters
            
        Returns:
            True if configuration was updated successfully
        """
        if not self._has_set_config:
            logger.warning("Plugin does not support runtime configuration")
            return False
        
        # Convert to C array
        param_array = (MVASConfigParam * len(params))()
        for i, (key, value) in enumerate(params.items()):
            param_array[i].key = key.encode('utf-8')
            param_array[i].value = str(value).encode('utf-8')
        
        result = self._lib.mvas_set_config(param_array, len(params))
        return result == 0
    
    def warmup(self, iterations: int = 10):
        """
        Warm up the plugin (pre-load GPU kernels).
        
        Args:
            iterations: Number of warmup iterations
        """
        if self._has_warmup:
            self._lib.mvas_warmup(iterations)
            logger.debug(f"Plugin warmed up with {iterations} iterations")
        else:
            # Manual warmup with dummy image
            info = self.get_info()
            dummy = np.random.randint(0, 255, (info.input_height, info.input_width, 3), dtype=np.uint8)
            for _ in range(iterations):
                try:
                    self.infer(dummy)
                except Exception:
                    break
            # Reset stats after warmup
            self._stats = PluginStats()
            self._stats.load_time_ms = self._stats.load_time_ms
    
    def get_stats(self) -> Dict[str, Any]:
        """Get plugin statistics"""
        return {
            "load_time_ms": self._stats.load_time_ms,
            "total_inferences": self._stats.total_inferences,
            "avg_inference_ms": self._stats.avg_inference_ms,
            "min_inference_ms": self._stats.min_inference_ms if self._stats.min_inference_ms != float('inf') else 0,
            "max_inference_ms": self._stats.max_inference_ms,
        }
    
    def cleanup(self):
        """Release resources and unload plugin"""
        if self._loaded:
            try:
                self._lib.mvas_cleanup()
            except Exception as e:
                logger.warning(f"Error during plugin cleanup: {e}")
            self._loaded = False
            logger.info(f"Plugin unloaded: {self.plugin_path.name}")
    
    def __del__(self):
        """Destructor - ensure cleanup"""
        self.cleanup()
    
    @property
    def is_loaded(self) -> bool:
        """Check if plugin is loaded and ready"""
        return self._loaded


# ============================================================================
# Plugin Manager
# ============================================================================

class PluginManager:
    """
    Manages multiple loaded native plugins.
    
    Features:
    - Load/unload plugins from .mvapp packages
    - Track plugin states and statistics
    - Provide unified interface for inference
    """
    
    def __init__(self, plugins_dir: Optional[Path] = None):
        """
        Initialize the plugin manager.
        
        Args:
            plugins_dir: Directory for extracting plugin packages
        """
        self.plugins_dir = plugins_dir or Path("plugins")
        self.plugins_dir.mkdir(parents=True, exist_ok=True)
        
        self._plugins: Dict[str, NativePlugin] = {}
        self._plugin_states: Dict[str, PluginState] = {}
        self._plugin_info: Dict[str, NativePluginInfo] = {}
    
    def load_plugin(
        self,
        app_id: str,
        plugin_path: Path,
        model_path: Path,
        config_json: str = "{}",
        deps_paths: Optional[list] = None
    ) -> NativePlugin:
        """
        Load a native plugin.
        
        Args:
            app_id: Unique application identifier
            plugin_path: Path to the DLL/SO file
            model_path: Path to the model file
            config_json: JSON configuration string
            deps_paths: Additional dependency paths
            
        Returns:
            Loaded NativePlugin instance
        """
        # Unload existing if present
        if app_id in self._plugins:
            self.unload_plugin(app_id)
        
        self._plugin_states[app_id] = PluginState.LOADING
        
        try:
            plugin = NativePlugin(plugin_path, model_path, config_json, deps_paths)
            self._plugins[app_id] = plugin
            self._plugin_info[app_id] = plugin.get_info()
            self._plugin_states[app_id] = PluginState.READY
            
            logger.info(f"Plugin loaded: {app_id} ({plugin.get_info().name})")
            return plugin
            
        except Exception as e:
            self._plugin_states[app_id] = PluginState.ERROR
            raise
    
    def get_plugin(self, app_id: str) -> Optional[NativePlugin]:
        """Get a loaded plugin by ID"""
        return self._plugins.get(app_id)
    
    def unload_plugin(self, app_id: str) -> bool:
        """Unload a plugin and release resources"""
        if app_id not in self._plugins:
            return False
        
        plugin = self._plugins.pop(app_id)
        plugin.cleanup()
        
        self._plugin_states[app_id] = PluginState.UNLOADED
        if app_id in self._plugin_info:
            del self._plugin_info[app_id]
        
        logger.info(f"Plugin unloaded: {app_id}")
        return True
    
    def list_plugins(self) -> list:
        """List all loaded plugins with their info"""
        result = []
        for app_id, plugin in self._plugins.items():
            info = plugin.get_info()
            stats = plugin.get_stats()
            result.append({
                "app_id": app_id,
                "name": info.name,
                "version": info.version,
                "state": self._plugin_states.get(app_id, PluginState.UNLOADED).value,
                "model_type": info.model_type,
                "load_time_ms": stats["load_time_ms"],
                "avg_inference_ms": stats["avg_inference_ms"],
            })
        return result
    
    def infer(
        self,
        app_id: str,
        image: np.ndarray,
        params: Optional[Dict[str, Any]] = None
    ) -> NativeInferenceResult:
        """
        Run inference using a loaded plugin.
        
        Args:
            app_id: Plugin/application ID
            image: Input image as numpy array
            params: Optional runtime parameters
            
        Returns:
            NativeInferenceResult with inference output
        """
        plugin = self._plugins.get(app_id)
        if plugin is None:
            raise ValueError(f"Plugin not loaded: {app_id}")
        
        if self._plugin_states.get(app_id) != PluginState.READY:
            raise RuntimeError(f"Plugin not ready: {app_id}")
        
        # Apply params if provided
        if params:
            plugin.set_config(params)
        
        return plugin.infer(image)
    
    def warmup_all(self, iterations: int = 10):
        """Warm up all loaded plugins"""
        for app_id, plugin in self._plugins.items():
            logger.info(f"Warming up plugin: {app_id}")
            plugin.warmup(iterations)
    
    def unload_all(self):
        """Unload all plugins"""
        for app_id in list(self._plugins.keys()):
            self.unload_plugin(app_id)
    
    def get_platform_binary_name(self) -> str:
        """Get the appropriate binary name for current platform"""
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
        else:
            return "linux-x64"


# Convenience function
def get_plugin_manager() -> PluginManager:
    """Get a global plugin manager instance"""
    if not hasattr(get_plugin_manager, "_instance"):
        get_plugin_manager._instance = PluginManager()
    return get_plugin_manager._instance

