"""
Adaptive Computer Vision for Different Client Sizes
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

from config.settings import CLIENT_DETECTION

logger = logging.getLogger(__name__)


class AdaptiveVision:
    """Computer vision system that adapts to different client sizes"""
    
    def __init__(self):
        self.base_size = CLIENT_DETECTION["client_size"]  # (765, 503)
        self.current_size: Optional[Tuple[int, int]] = None
        self.scale_factor_x = 1.0
        self.scale_factor_y = 1.0
        
    def set_client_size(self, width: int, height: int):
        """Set current client size and calculate scale factors"""
        self.current_size = (width, height)
        
        # Calculate scale factors
        self.scale_factor_x = width / self.base_size[0]
        self.scale_factor_y = height / self.base_size[1]
        
        logger.info(f"Adaptive vision configured for {width}x{height}")
        logger.info(f"Scale factors: X={self.scale_factor_x:.2f}, Y={self.scale_factor_y:.2f}")
        
        # Update settings with calculated scale factor
        CLIENT_DETECTION["scale_factor"] = min(self.scale_factor_x, self.scale_factor_y)
    
    def scale_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """Scale coordinates from base size to current size"""
        scaled_x = int(x * self.scale_factor_x)
        scaled_y = int(y * self.scale_factor_y)
        return (scaled_x, scaled_y)
    
    def scale_template(self, template: np.ndarray) -> np.ndarray:
        """Scale template to match current client size"""
        if self.current_size is None:
            return template
        
        # Calculate new template size
        old_height, old_width = template.shape[:2]
        new_width = int(old_width * self.scale_factor_x)
        new_height = int(old_height * self.scale_factor_y)
        
        # Resize template
        scaled_template = cv2.resize(template, (new_width, new_height), 
                                   interpolation=cv2.INTER_AREA)
        
        return scaled_template
    
    def scale_region(self, region: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """Scale region coordinates (x, y, width, height)"""
        x, y, w, h = region
        
        scaled_x = int(x * self.scale_factor_x)
        scaled_y = int(y * self.scale_factor_y)
        scaled_w = int(w * self.scale_factor_x)
        scaled_h = int(h * self.scale_factor_y)
        
        return (scaled_x, scaled_y, scaled_w, scaled_h)
    
    def is_large_client(self) -> bool:
        """Check if client is significantly larger than standard"""
        if self.current_size is None:
            return False
        
        return (self.current_size[0] > 1200 or 
                self.current_size[1] > 800 or
                self.scale_factor_x > 2.0 or 
                self.scale_factor_y > 2.0)
    
    def get_performance_impact(self) -> str:
        """Get performance impact assessment"""
        if self.current_size is None:
            return "Unknown"
        
        total_pixels = self.current_size[0] * self.current_size[1]
        base_pixels = self.base_size[0] * self.base_size[1]
        pixel_ratio = total_pixels / base_pixels
        
        if pixel_ratio > 16:
            return "Very High (Consider reducing client size)"
        elif pixel_ratio > 9:
            return "High (May impact performance)"
        elif pixel_ratio > 4:
            return "Medium (Acceptable performance)"
        elif pixel_ratio > 1.5:
            return "Low (Good performance)"
        else:
            return "Minimal (Excellent performance)"
    
    def get_recommendations(self) -> list:
        """Get recommendations for optimal performance"""
        recommendations = []
        
        if self.current_size is None:
            return ["Run client calibration first"]
        
        if self.is_large_client():
            recommendations.extend([
                "Consider using a smaller client window for better performance",
                "Large clients work but may be slower",
                "Templates may need to be recreated for this size"
            ])
        
        if self.scale_factor_x != self.scale_factor_y:
            recommendations.append("Client aspect ratio differs from standard - some features may need adjustment")
        
        if not recommendations:
            recommendations.append("Client size is optimal for performance")
        
        return recommendations


# Global adaptive vision instance
adaptive_vision = AdaptiveVision() 