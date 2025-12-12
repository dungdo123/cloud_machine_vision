"""
MVAS Preprocessing Pipeline

Configurable image preprocessing pipeline.
"""

import logging
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np

from ..models import TransformPipeline, TransformOp
from .transforms import create_transform, Transform

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Configurable image preprocessing pipeline.
    
    Applies a sequence of transforms to prepare images for inference.
    
    Features:
    - Configurable via JSON/dict
    - Optional ROI masking
    - Automatic color mode conversion
    - Caching of static transforms
    """
    
    def __init__(
        self,
        pipeline: TransformPipeline = None,
        target_size: Tuple[int, int] = (256, 256),
        color_mode: str = "RGB",
        roi_mask: Optional[np.ndarray] = None,
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            pipeline: Transform pipeline configuration
            target_size: Target (width, height) for resize
            color_mode: Target color mode ("RGB" or "BGR")
            roi_mask: Optional ROI mask to apply
        """
        self.target_size = target_size
        self.color_mode = color_mode
        self.roi_mask = roi_mask
        
        # Build transform list
        self.transforms: List[Transform] = []
        
        if pipeline and pipeline.pipeline:
            for op in pipeline.pipeline:
                if not op.enabled:
                    continue
                try:
                    transform = create_transform(op.op, op.params)
                    self.transforms.append(transform)
                except Exception as e:
                    logger.warning(f"Failed to create transform '{op.op}': {e}")
        
        # Add default transforms if pipeline is empty
        if not self.transforms:
            self._add_default_transforms()
        
        logger.debug(f"Preprocessing pipeline: {[t.name for t in self.transforms]}")
    
    def _add_default_transforms(self):
        """Add default transforms for anomaly detection"""
        from .transforms import Resize, BGR2RGB, Normalize, ToTensor, AddBatchDim
        
        # Default pipeline: resize -> BGR2RGB -> normalize -> to_tensor -> add_batch
        self.transforms = [
            Resize({"width": self.target_size[0], "height": self.target_size[1]}),
            BGR2RGB({}),
            Normalize({
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "scale": 255.0
            }),
            ToTensor({"dtype": "float32", "channel_order": "CHW"}),
            AddBatchDim({}),
        ]
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process an image through the pipeline.
        
        Args:
            image: Input image (HWC format, typically BGR from OpenCV)
            
        Returns:
            Processed tensor ready for inference (NCHW format)
        """
        if image is None:
            raise ValueError("Input image is None")
        
        # Make a copy to avoid modifying original
        result = image.copy()
        
        # Apply ROI mask if available
        if self.roi_mask is not None:
            result = self._apply_roi_mask(result)
        
        # Apply transforms
        for transform in self.transforms:
            try:
                result = transform(result)
            except Exception as e:
                logger.error(f"Transform '{transform.name}' failed: {e}")
                raise
        
        return result
    
    def _apply_roi_mask(self, image: np.ndarray) -> np.ndarray:
        """Apply ROI mask to image"""
        if self.roi_mask is None:
            return image
        
        mask = self.roi_mask
        
        # Resize mask to match image if needed
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Normalize mask to [0, 1]
        if mask.max() > 1:
            mask = mask.astype(np.float32) / 255.0
        
        # Apply mask
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
        
        return (image * mask).astype(image.dtype)
    
    def get_info(self) -> Dict[str, Any]:
        """Get pipeline information"""
        return {
            "target_size": self.target_size,
            "color_mode": self.color_mode,
            "has_roi_mask": self.roi_mask is not None,
            "transforms": [t.name for t in self.transforms],
        }
    
    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "PreprocessingPipeline":
        """
        Create pipeline from dictionary configuration.
        
        Args:
            config: Dictionary with pipeline configuration
            
        Returns:
            PreprocessingPipeline instance
        """
        pipeline = None
        if "pipeline" in config:
            pipeline = TransformPipeline(
                pipeline=[TransformOp(**op) for op in config["pipeline"]]
            )
        
        return cls(
            pipeline=pipeline,
            target_size=tuple(config.get("target_size", (256, 256))),
            color_mode=config.get("color_mode", "RGB"),
        )
    
    @classmethod
    def from_json(cls, json_path: str) -> "PreprocessingPipeline":
        """
        Load pipeline from JSON file.
        
        Args:
            json_path: Path to JSON configuration file
            
        Returns:
            PreprocessingPipeline instance
        """
        import json
        
        with open(json_path, 'r') as f:
            config = json.load(f)
        
        return cls.from_dict(config)

