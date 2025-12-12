"""
MVAS Visualization

Generate visualizations for inspection results.
"""

import logging
from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Visualizer:
    """
    Generate visualizations for inspection results.
    
    Supports:
    - Anomaly heatmap overlay
    - Bounding boxes for defects
    - Decision text overlay
    - Color-coded borders
    """
    
    # Default colors (BGR)
    COLORS = {
        "pass": (0, 255, 0),      # Green
        "fail": (0, 0, 255),      # Red
        "review": (0, 165, 255),  # Orange
        "error": (128, 128, 128), # Gray
    }
    
    # Colormaps for heatmaps
    COLORMAPS = {
        "jet": cv2.COLORMAP_JET,
        "hot": cv2.COLORMAP_HOT,
        "rainbow": cv2.COLORMAP_RAINBOW,
        "turbo": cv2.COLORMAP_TURBO,
        "inferno": cv2.COLORMAP_INFERNO,
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or {}
        
        self.heatmap_colormap = self.COLORMAPS.get(
            self.config.get("heatmap_colormap", "jet"),
            cv2.COLORMAP_JET
        )
        self.heatmap_alpha = self.config.get("heatmap_alpha", 0.5)
        self.bbox_thickness = self.config.get("bbox_thickness", 2)
        self.border_thickness = self.config.get("border_thickness", 10)
        self.font_scale = self.config.get("font_scale", 1.0)
    
    def draw_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: Optional[float] = None,
    ) -> np.ndarray:
        """
        Overlay anomaly heatmap on image.
        
        Args:
            image: Original image (BGR)
            heatmap: Anomaly map (single channel, values 0-1 or 0-255)
            alpha: Overlay alpha (default from config)
            
        Returns:
            Image with heatmap overlay
        """
        alpha = alpha or self.heatmap_alpha
        
        # Ensure heatmap is 2D
        if len(heatmap.shape) == 3:
            heatmap = heatmap.squeeze()
        if len(heatmap.shape) == 4:
            heatmap = heatmap.squeeze()
        
        # Normalize to 0-255
        if heatmap.max() <= 1.0:
            heatmap = (heatmap * 255).astype(np.uint8)
        else:
            heatmap = heatmap.astype(np.uint8)
        
        # Resize heatmap to match image
        if heatmap.shape[:2] != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap, self.heatmap_colormap)
        
        # Blend with original image
        result = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
        
        return result
    
    def draw_bounding_boxes(
        self,
        image: np.ndarray,
        boxes: list,
        color: Tuple[int, int, int] = None,
        thickness: int = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Image to draw on
            boxes: List of boxes as (x, y, w, h) or (x1, y1, x2, y2)
            color: Box color (BGR)
            thickness: Line thickness
            
        Returns:
            Image with boxes drawn
        """
        result = image.copy()
        color = color or self.COLORS["fail"]
        thickness = thickness or self.bbox_thickness
        
        for box in boxes:
            if len(box) == 4:
                x1, y1, x2, y2 = box
                # Check if (x, y, w, h) format
                if x2 < x1 or y2 < y1:
                    x1, y1, w, h = box
                    x2, y2 = x1 + w, y1 + h
                
                cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), 
                             color, thickness)
        
        return result
    
    def draw_decision_border(
        self,
        image: np.ndarray,
        decision: str,
        thickness: int = None,
    ) -> np.ndarray:
        """
        Draw color-coded border based on decision.
        
        Args:
            image: Image to draw on
            decision: Decision string ("pass", "fail", "review")
            thickness: Border thickness
            
        Returns:
            Image with border
        """
        result = image.copy()
        color = self.COLORS.get(decision, self.COLORS["error"])
        thickness = thickness or self.border_thickness
        
        h, w = result.shape[:2]
        
        # Draw border
        cv2.rectangle(result, (0, 0), (w-1, h-1), color, thickness)
        
        return result
    
    def draw_text(
        self,
        image: np.ndarray,
        text: str,
        position: Tuple[int, int] = (10, 30),
        color: Tuple[int, int, int] = (255, 255, 255),
        background: bool = True,
    ) -> np.ndarray:
        """
        Draw text on image.
        
        Args:
            image: Image to draw on
            text: Text to draw
            position: Text position (x, y)
            color: Text color
            background: Draw background rectangle
            
        Returns:
            Image with text
        """
        result = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self.font_scale
        thickness = 2
        
        # Get text size
        (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
        
        x, y = position
        
        # Draw background
        if background:
            cv2.rectangle(
                result,
                (x - 5, y - text_h - 5),
                (x + text_w + 5, y + baseline + 5),
                (0, 0, 0),
                -1
            )
        
        # Draw text
        cv2.putText(result, text, (x, y), font, scale, color, thickness)
        
        return result
    
    def create_result_visualization(
        self,
        image: np.ndarray,
        decision: str,
        anomaly_score: float,
        anomaly_map: Optional[np.ndarray] = None,
        bounding_boxes: Optional[list] = None,
    ) -> np.ndarray:
        """
        Create complete result visualization.
        
        Args:
            image: Original image
            decision: Decision result
            anomaly_score: Anomaly score
            anomaly_map: Optional anomaly heatmap
            bounding_boxes: Optional bounding boxes
            
        Returns:
            Visualization image
        """
        result = image.copy()
        
        # Draw heatmap overlay
        if anomaly_map is not None:
            result = self.draw_heatmap(result, anomaly_map)
        
        # Draw bounding boxes
        if bounding_boxes:
            result = self.draw_bounding_boxes(result, bounding_boxes)
        
        # Draw decision border
        result = self.draw_decision_border(result, decision)
        
        # Draw text overlay
        color = self.COLORS.get(decision, self.COLORS["error"])
        text = f"{decision.upper()} ({anomaly_score:.2f})"
        result = self.draw_text(result, text, (10, 30), color)
        
        return result
    
    def create_comparison(
        self,
        original: np.ndarray,
        visualization: np.ndarray,
        horizontal: bool = True,
    ) -> np.ndarray:
        """
        Create side-by-side comparison.
        
        Args:
            original: Original image
            visualization: Visualization image
            horizontal: Stack horizontally (True) or vertically (False)
            
        Returns:
            Combined image
        """
        # Ensure same size
        if original.shape != visualization.shape:
            visualization = cv2.resize(
                visualization,
                (original.shape[1], original.shape[0])
            )
        
        if horizontal:
            return np.hstack([original, visualization])
        else:
            return np.vstack([original, visualization])
    
    def save_visualization(
        self,
        image: np.ndarray,
        path: str,
        quality: int = 85,
    ) -> bool:
        """
        Save visualization to file.
        
        Args:
            image: Image to save
            path: Output path
            quality: JPEG quality (1-100)
            
        Returns:
            True if saved successfully
        """
        try:
            if path.lower().endswith(".jpg") or path.lower().endswith(".jpeg"):
                cv2.imwrite(path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(path, image)
            return True
        except Exception as e:
            logger.error(f"Failed to save visualization: {e}")
            return False

