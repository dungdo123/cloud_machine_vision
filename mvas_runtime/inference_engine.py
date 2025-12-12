"""
MVAS Inference Engine

Multi-backend inference engine supporting ONNX, TorchScript, TensorRT, and OpenVINO.
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from .models import ModelFramework, ModelType, InferenceOutput

logger = logging.getLogger(__name__)


class InferenceBackend(ABC):
    """Abstract base class for inference backends"""
    
    @abstractmethod
    def load(self, model_path: Path, config: Dict[str, Any]) -> None:
        """Load the model"""
        pass
    
    @abstractmethod
    def infer(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on input tensor"""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources"""
        pass
    
    @abstractmethod
    def get_input_shape(self) -> List[int]:
        """Get expected input shape"""
        pass
    
    @abstractmethod
    def get_output_names(self) -> List[str]:
        """Get output tensor names"""
        pass


class ONNXBackend(InferenceBackend):
    """ONNX Runtime backend"""
    
    def __init__(self):
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
    
    def load(self, model_path: Path, config: Dict[str, Any]) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime-gpu")
        
        # Configure providers
        device = config.get("device", "auto")
        providers = self._get_providers(device)
        
        # Session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        if "num_threads" in config:
            sess_options.intra_op_num_threads = config["num_threads"]
        
        # Create session
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [o.name for o in self.session.get_outputs()]
        
        logger.info(f"ONNX model loaded. Input: {self.input_name}, Shape: {self.input_shape}")
        logger.info(f"Using provider: {self.session.get_providers()}")
    
    def _get_providers(self, device: str) -> List[str]:
        if device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cpu":
            return ["CPUExecutionProvider"]
        else:  # auto
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    
    def infer(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return {name: output for name, output in zip(self.output_names, outputs)}
    
    def cleanup(self) -> None:
        self.session = None
    
    def get_input_shape(self) -> List[int]:
        return list(self.input_shape)
    
    def get_output_names(self) -> List[str]:
        return self.output_names


class TorchScriptBackend(InferenceBackend):
    """TorchScript (LibTorch) backend"""
    
    def __init__(self):
        self.model = None
        self.device = None
        self.input_shape = None
    
    def load(self, model_path: Path, config: Dict[str, Any]) -> None:
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch not installed. Install with: pip install torch")
        
        # Determine device
        device_str = config.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        
        # Load model
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()
        
        # Apply precision settings
        precision = config.get("precision", "fp32")
        if precision == "fp16" and self.device.type == "cuda":
            self.model = self.model.half()
        
        logger.info(f"TorchScript model loaded on {self.device}")
    
    def infer(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        import torch
        
        with torch.no_grad():
            tensor = torch.from_numpy(input_tensor).to(self.device)
            
            # Handle precision
            if next(self.model.parameters()).dtype == torch.float16:
                tensor = tensor.half()
            
            output = self.model(tensor)
            
            # Handle different output types
            if isinstance(output, torch.Tensor):
                return {"output": output.cpu().numpy()}
            elif isinstance(output, (tuple, list)):
                return {f"output_{i}": o.cpu().numpy() for i, o in enumerate(output)}
            elif isinstance(output, dict):
                return {k: v.cpu().numpy() for k, v in output.items()}
            else:
                return {"output": output}
    
    def cleanup(self) -> None:
        import torch
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_input_shape(self) -> List[int]:
        return self.input_shape or [1, 3, 256, 256]
    
    def get_output_names(self) -> List[str]:
        return ["output"]


class TensorRTBackend(InferenceBackend):
    """TensorRT backend (NVIDIA)"""
    
    def __init__(self):
        self.engine = None
        self.context = None
    
    def load(self, model_path: Path, config: Dict[str, Any]) -> None:
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            raise ImportError("TensorRT not installed. Install NVIDIA TensorRT SDK.")
        
        # Load engine
        logger.info("TensorRT backend not fully implemented yet")
        raise NotImplementedError("TensorRT backend is not yet implemented")
    
    def infer(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        raise NotImplementedError()
    
    def cleanup(self) -> None:
        pass
    
    def get_input_shape(self) -> List[int]:
        return []
    
    def get_output_names(self) -> List[str]:
        return []


class OpenVINOBackend(InferenceBackend):
    """OpenVINO backend (Intel)"""
    
    def __init__(self):
        self.model = None
        self.compiled_model = None
        self.infer_request = None
    
    def load(self, model_path: Path, config: Dict[str, Any]) -> None:
        try:
            from openvino.runtime import Core
        except ImportError:
            raise ImportError("OpenVINO not installed. Install with: pip install openvino")
        
        core = Core()
        
        # Read model
        self.model = core.read_model(str(model_path))
        
        # Compile for target device
        device = config.get("device", "CPU")
        if device == "auto":
            device = "CPU"
        
        self.compiled_model = core.compile_model(self.model, device)
        self.infer_request = self.compiled_model.create_infer_request()
        
        logger.info(f"OpenVINO model loaded on {device}")
    
    def infer(self, input_tensor: np.ndarray) -> Dict[str, np.ndarray]:
        self.infer_request.infer({0: input_tensor})
        
        outputs = {}
        for i, output in enumerate(self.compiled_model.outputs):
            outputs[output.any_name] = self.infer_request.get_output_tensor(i).data.copy()
        
        return outputs
    
    def cleanup(self) -> None:
        self.infer_request = None
        self.compiled_model = None
        self.model = None
    
    def get_input_shape(self) -> List[int]:
        if self.model:
            return list(self.model.inputs[0].shape)
        return []
    
    def get_output_names(self) -> List[str]:
        if self.model:
            return [o.any_name for o in self.model.outputs]
        return []


class InferenceEngine:
    """
    Main inference engine with multi-backend support.
    
    Automatically selects and configures the appropriate backend
    based on the model format and available hardware.
    """
    
    BACKEND_MAP = {
        ModelFramework.ONNX: ONNXBackend,
        ModelFramework.TORCHSCRIPT: TorchScriptBackend,
        ModelFramework.TENSORRT: TensorRTBackend,
        ModelFramework.OPENVINO: OpenVINOBackend,
    }
    
    def __init__(
        self,
        model_path: Path,
        framework: ModelFramework,
        model_type: ModelType,
        config: Dict[str, Any] = None,
        additional_files: Dict[str, Path] = None,
    ):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the model file
            framework: Model framework (ONNX, TorchScript, etc.)
            model_type: Type of model (anomaly detection, classification, etc.)
            config: Runtime configuration
            additional_files: Additional files needed (e.g., memory bank for PatchCore)
        """
        self.model_path = model_path
        self.framework = framework
        self.model_type = model_type
        self.config = config or {}
        self.additional_files = additional_files or {}
        
        # Initialize backend
        backend_class = self.BACKEND_MAP.get(framework)
        if backend_class is None:
            raise ValueError(f"Unsupported framework: {framework}")
        
        self.backend = backend_class()
        self._load_model()
        
        # Warmup
        warmup_iters = self.config.get("warmup_iterations", 3)
        self._warmup(warmup_iters)
    
    def _load_model(self):
        """Load the model into the backend"""
        logger.info(f"Loading model: {self.model_path}")
        self.backend.load(self.model_path, self.config)
    
    def _warmup(self, iterations: int = 3):
        """Warmup the model with dummy inputs"""
        if iterations <= 0:
            return
        
        logger.info(f"Warming up model with {iterations} iterations...")
        
        # Create dummy input
        input_shape = self.backend.get_input_shape()
        if input_shape and all(isinstance(d, int) and d > 0 for d in input_shape):
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
        else:
            # Default shape
            dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
        
        for i in range(iterations):
            try:
                self.backend.infer(dummy_input)
            except Exception as e:
                logger.warning(f"Warmup iteration {i} failed: {e}")
                break
        
        logger.info("Model warmup complete")
    
    def infer(self, input_tensor: np.ndarray) -> InferenceOutput:
        """
        Run inference on the input tensor.
        
        Args:
            input_tensor: Preprocessed input tensor (NCHW format)
            
        Returns:
            InferenceOutput with results
        """
        start_time = time.perf_counter()
        
        # Run inference
        raw_outputs = self.backend.infer(input_tensor)
        
        inference_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Process outputs based on model type
        output = self._process_outputs(raw_outputs)
        output.inference_time_ms = inference_time_ms
        
        return output
    
    def _process_outputs(self, raw_outputs: Dict[str, np.ndarray]) -> InferenceOutput:
        """Process raw outputs based on model type"""
        
        if self.model_type == ModelType.ANOMALY_DETECTION:
            return self._process_anomaly_outputs(raw_outputs)
        elif self.model_type == ModelType.CLASSIFICATION:
            return self._process_classification_outputs(raw_outputs)
        elif self.model_type == ModelType.OBJECT_DETECTION:
            return self._process_detection_outputs(raw_outputs)
        elif self.model_type == ModelType.SEGMENTATION:
            return self._process_segmentation_outputs(raw_outputs)
        else:
            # Generic output
            return InferenceOutput(
                anomaly_score=0.0,
            )
    
    def _process_anomaly_outputs(self, raw_outputs: Dict[str, np.ndarray]) -> InferenceOutput:
        """Process anomaly detection outputs"""
        # Try to find anomaly score and map
        anomaly_score = 0.0
        anomaly_map = None
        
        # Common output names
        score_names = ["anomaly_score", "score", "output_0", "output"]
        map_names = ["anomaly_map", "heatmap", "output_1", "segmentation"]
        
        for name in score_names:
            if name in raw_outputs:
                score_data = raw_outputs[name]
                if score_data.size == 1:
                    anomaly_score = float(score_data.flatten()[0])
                else:
                    anomaly_score = float(np.max(score_data))
                break
        
        for name in map_names:
            if name in raw_outputs:
                anomaly_map = raw_outputs[name]
                # If anomaly_score not found, derive from map
                if anomaly_score == 0.0:
                    anomaly_score = float(np.max(anomaly_map))
                break
        
        # If only one output, use it for both
        if len(raw_outputs) == 1:
            output = list(raw_outputs.values())[0]
            if output.size == 1:
                anomaly_score = float(output.flatten()[0])
            else:
                anomaly_map = output
                anomaly_score = float(np.max(output))
        
        return InferenceOutput(
            anomaly_score=anomaly_score,
            anomaly_map=anomaly_map,
        )
    
    def _process_classification_outputs(self, raw_outputs: Dict[str, np.ndarray]) -> InferenceOutput:
        """Process classification outputs"""
        # Get logits/probabilities
        output = list(raw_outputs.values())[0]
        
        # Apply softmax if needed
        if output.max() > 1.0 or output.min() < 0.0:
            exp_output = np.exp(output - np.max(output))
            probs = exp_output / exp_output.sum()
        else:
            probs = output
        
        probs = probs.flatten()
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        
        class_scores = {str(i): float(p) for i, p in enumerate(probs)}
        
        return InferenceOutput(
            anomaly_score=1.0 - confidence,  # Higher score = less confident
            class_scores=class_scores,
            class_label=str(class_idx),
        )
    
    def _process_detection_outputs(self, raw_outputs: Dict[str, np.ndarray]) -> InferenceOutput:
        """Process object detection outputs"""
        # This is a simplified version - actual implementation depends on model
        detections = []
        
        return InferenceOutput(
            anomaly_score=0.0,
            detections=detections,
        )
    
    def _process_segmentation_outputs(self, raw_outputs: Dict[str, np.ndarray]) -> InferenceOutput:
        """Process segmentation outputs"""
        output = list(raw_outputs.values())[0]
        
        return InferenceOutput(
            anomaly_score=0.0,
            anomaly_map=output,
        )
    
    def cleanup(self):
        """Clean up resources"""
        if self.backend:
            self.backend.cleanup()
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_path": str(self.model_path),
            "framework": self.framework.value,
            "model_type": self.model_type.value,
            "input_shape": self.backend.get_input_shape(),
            "output_names": self.backend.get_output_names(),
        }

