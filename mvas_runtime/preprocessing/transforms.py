"""
MVAS Image Transforms

Individual transform operations for the preprocessing pipeline.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Transform(ABC):
    """Base class for image transforms"""
    
    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
    
    @abstractmethod
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply transform to image"""
        pass
    
    @property
    def name(self) -> str:
        return self.__class__.__name__


class Resize(Transform):
    """Resize image to target dimensions"""
    
    INTERPOLATION_MAP = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "area": cv2.INTER_AREA,
    }
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        width = self.params.get("width", 256)
        height = self.params.get("height", 256)
        interp = self.params.get("interpolation", "bilinear")
        keep_aspect = self.params.get("keep_aspect", False)
        
        interp_flag = self.INTERPOLATION_MAP.get(interp, cv2.INTER_LINEAR)
        
        if keep_aspect:
            # Resize maintaining aspect ratio, then pad
            h, w = image.shape[:2]
            scale = min(width / w, height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            
            image = cv2.resize(image, (new_w, new_h), interpolation=interp_flag)
            
            # Pad to target size
            pad_w = (width - new_w) // 2
            pad_h = (height - new_h) // 2
            image = cv2.copyMakeBorder(
                image,
                pad_h, height - new_h - pad_h,
                pad_w, width - new_w - pad_w,
                cv2.BORDER_CONSTANT,
                value=self.params.get("pad_value", [0, 0, 0])
            )
        else:
            image = cv2.resize(image, (width, height), interpolation=interp_flag)
        
        return image


class CropCenter(Transform):
    """Crop center region of image"""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        width = self.params.get("width", 224)
        height = self.params.get("height", 224)
        
        h, w = image.shape[:2]
        
        start_x = (w - width) // 2
        start_y = (h - height) // 2
        
        return image[start_y:start_y+height, start_x:start_x+width]


class CropROI(Transform):
    """Crop to region of interest"""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        x = self.params.get("x", 0)
        y = self.params.get("y", 0)
        width = self.params.get("width", image.shape[1])
        height = self.params.get("height", image.shape[0])
        
        return image[y:y+height, x:x+width]


class Normalize(Transform):
    """Normalize image with mean and std"""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        mean = np.array(self.params.get("mean", [0.485, 0.456, 0.406]))
        std = np.array(self.params.get("std", [0.229, 0.224, 0.225]))
        scale = self.params.get("scale", 255.0)
        
        # Convert to float and scale
        image = image.astype(np.float32) / scale
        
        # Normalize
        image = (image - mean) / std
        
        return image.astype(np.float32)


class ToTensor(Transform):
    """Convert image to tensor format (CHW)"""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        dtype = self.params.get("dtype", "float32")
        channel_order = self.params.get("channel_order", "CHW")
        
        # Convert dtype
        if dtype == "float32":
            image = image.astype(np.float32)
        elif dtype == "float16":
            image = image.astype(np.float16)
        
        # Transpose to CHW if needed
        if channel_order == "CHW" and len(image.shape) == 3:
            image = np.transpose(image, (2, 0, 1))
        
        return image


class AddBatchDim(Transform):
    """Add batch dimension"""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return np.expand_dims(image, axis=0)


class BGR2RGB(Transform):
    """Convert BGR to RGB"""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class RGB2BGR(Transform):
    """Convert RGB to BGR"""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


class Grayscale(Transform):
    """Convert to grayscale"""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        keep_dims = self.params.get("keep_dims", True)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if keep_dims:
                gray = np.expand_dims(gray, axis=-1)
            return gray
        return image


class GaussianBlur(Transform):
    """Apply Gaussian blur"""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        ksize = self.params.get("kernel_size", 5)
        sigma = self.params.get("sigma", 0)
        
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)


class CLAHE(Transform):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)"""
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        clip_limit = self.params.get("clip_limit", 2.0)
        tile_grid = self.params.get("tile_grid_size", (8, 8))
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        
        if len(image.shape) == 3:
            # Apply to L channel of LAB
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            return clahe.apply(image)


class ApplyMask(Transform):
    """Apply ROI mask to image"""
    
    def __call__(self, image: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        if mask is None:
            mask = self.params.get("mask")
        
        if mask is None:
            return image
        
        # Ensure mask is same size as image
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Apply mask
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=-1)
        
        return image * (mask / 255.0)


# Transform registry
TRANSFORM_REGISTRY: Dict[str, type] = {
    "resize": Resize,
    "crop_center": CropCenter,
    "crop_roi": CropROI,
    "normalize": Normalize,
    "to_tensor": ToTensor,
    "add_batch_dim": AddBatchDim,
    "bgr2rgb": BGR2RGB,
    "rgb2bgr": RGB2BGR,
    "grayscale": Grayscale,
    "gaussian_blur": GaussianBlur,
    "clahe": CLAHE,
    "apply_mask": ApplyMask,
}


def create_transform(op: str, params: Dict[str, Any] = None) -> Transform:
    """
    Create a transform by name.
    
    Args:
        op: Transform operation name
        params: Transform parameters
        
    Returns:
        Transform instance
        
    Raises:
        ValueError: If transform is not found
    """
    if op not in TRANSFORM_REGISTRY:
        raise ValueError(f"Unknown transform: {op}. Available: {list(TRANSFORM_REGISTRY.keys())}")
    
    return TRANSFORM_REGISTRY[op](params or {})

