"""
Computer vision utilities for OSRS Bot Framework
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging

from config.settings import VISION, TEMPLATES_DIR, DEVELOPMENT

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a detected object or template match"""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    label: str = ""
    
    @property
    def center(self) -> Tuple[int, int]:
        """Return center coordinates"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """Return bounding box as (x, y, w, h)"""
        return (self.x, self.y, self.width, self.height)


class TemplateManager:
    """Manages template images for matching"""
    
    def __init__(self):
        self.templates: Dict[str, np.ndarray] = {}
        self.template_sizes: Dict[str, Tuple[int, int]] = {}
        self.load_templates()
    
    def load_templates(self) -> None:
        """Load all templates from the templates directory"""
        templates_path = Path(TEMPLATES_DIR)
        if not templates_path.exists():
            logger.warning(f"Templates directory not found: {templates_path}")
            return
        
        for template_file in templates_path.glob("*.png"):
            try:
                template = cv2.imread(str(template_file), cv2.IMREAD_COLOR)
                if template is not None:
                    name = template_file.stem
                    self.templates[name] = template
                    self.template_sizes[name] = (template.shape[1], template.shape[0])
                    logger.debug(f"Loaded template: {name} ({template.shape[1]}x{template.shape[0]})")
                else:
                    logger.warning(f"Failed to load template: {template_file}")
            except Exception as e:
                logger.error(f"Error loading template {template_file}: {e}")
        
        logger.info(f"Loaded {len(self.templates)} templates")
    
    def add_template(self, name: str, image: np.ndarray) -> None:
        """Add a template image"""
        self.templates[name] = image.copy()
        self.template_sizes[name] = (image.shape[1], image.shape[0])
        logger.debug(f"Added template: {name}")
    
    def get_template(self, name: str) -> Optional[np.ndarray]:
        """Get template by name"""
        return self.templates.get(name)
    
    def save_template(self, name: str, image: np.ndarray) -> bool:
        """Save template to disk"""
        try:
            template_path = Path(TEMPLATES_DIR) / f"{name}.png"
            cv2.imwrite(str(template_path), image)
            self.add_template(name, image)
            logger.info(f"Saved template: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to save template {name}: {e}")
            return False


class TemplateMatching:
    """Template matching functionality"""
    
    def __init__(self, template_manager: TemplateManager):
        self.template_manager = template_manager
        self.threshold = VISION["template_matching"]["threshold"]
        self.method = getattr(cv2, VISION["template_matching"]["method"])
        self.max_matches = VISION["template_matching"]["max_matches"]
    
    def find_template(self, image: np.ndarray, template_name: str, 
                     threshold: Optional[float] = None, 
                     region: Optional[Tuple[int, int, int, int]] = None) -> List[Detection]:
        """
        Find template matches in image
        Returns list of Detection objects
        """
        template = self.template_manager.get_template(template_name)
        if template is None:
            logger.warning(f"Template not found: {template_name}")
            return []
        
        return self.match_template(image, template, template_name, threshold, region)
    
    def match_template(self, image: np.ndarray, template: np.ndarray, 
                      label: str = "", threshold: Optional[float] = None,
                      region: Optional[Tuple[int, int, int, int]] = None) -> List[Detection]:
        """
        Perform template matching
        """
        try:
            # Use region of interest if specified
            search_image = image
            offset_x, offset_y = 0, 0
            
            if region:
                x, y, w, h = region
                search_image = image[y:y+h, x:x+w]
                offset_x, offset_y = x, y
            
            # Perform template matching
            result = cv2.matchTemplate(search_image, template, self.method)
            
            # Use provided threshold or default
            match_threshold = threshold or self.threshold
            
            # Find matches above threshold
            locations = np.where(result >= match_threshold)
            matches = []
            
            template_h, template_w = template.shape[:2]
            
            # Convert locations to Detection objects
            for pt in zip(*locations[::-1]):  # Switch x,y coordinates
                confidence = result[pt[1], pt[0]]
                detection = Detection(
                    x=pt[0] + offset_x,
                    y=pt[1] + offset_y,
                    width=template_w,
                    height=template_h,
                    confidence=float(confidence),
                    label=label
                )
                matches.append(detection)
            
            # Remove overlapping matches (Non-Maximum Suppression)
            if len(matches) > 1:
                matches = self._nms(matches)
            
            # Limit number of matches
            matches = matches[:self.max_matches]
            
            logger.debug(f"Found {len(matches)} matches for template '{label}'")
            return matches
            
        except Exception as e:
            logger.error(f"Template matching failed: {e}")
            return []
    
    def _nms(self, detections: List[Detection], overlap_threshold: float = 0.3) -> List[Detection]:
        """Non-Maximum Suppression to remove overlapping detections"""
        if not detections:
            return []
        
        # Sort by confidence (descending)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        keep = []
        while detections:
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            detections = [d for d in detections if self._iou(current, d) < overlap_threshold]
        
        return keep
    
    def _iou(self, det1: Detection, det2: Detection) -> float:
        """Calculate Intersection over Union (IoU) between two detections"""
        x1 = max(det1.x, det2.x)
        y1 = max(det1.y, det2.y)
        x2 = min(det1.x + det1.width, det2.x + det2.width)
        y2 = min(det1.y + det1.height, det2.y + det2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = det1.width * det1.height
        area2 = det2.width * det2.height
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class ColorDetection:
    """Color-based object detection"""
    
    def __init__(self):
        self.hsv_tolerance = VISION["color_detection"]["hsv_tolerance"]
        self.min_area = VISION["color_detection"]["min_area"]
    
    def find_color(self, image: np.ndarray, color_hsv: Tuple[int, int, int],
                   tolerance: Optional[int] = None,
                   region: Optional[Tuple[int, int, int, int]] = None) -> List[Detection]:
        """
        Find objects of specific color in HSV space
        """
        try:
            # Use region of interest if specified
            search_image = image
            offset_x, offset_y = 0, 0
            
            if region:
                x, y, w, h = region
                search_image = image[y:y+h, x:x+w]
                offset_x, offset_y = x, y
            
            # Convert to HSV
            hsv = cv2.cvtColor(search_image, cv2.COLOR_BGR2HSV)
            
            # Create color mask
            tol = tolerance or self.hsv_tolerance
            lower_bound = np.array([max(0, color_hsv[0] - tol),
                                   max(0, color_hsv[1] - tol),
                                   max(0, color_hsv[2] - tol)])
            upper_bound = np.array([min(179, color_hsv[0] + tol),
                                   min(255, color_hsv[1] + tol),
                                   min(255, color_hsv[2] + tol)])
            
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            detections = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.min_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    detection = Detection(
                        x=x + offset_x,
                        y=y + offset_y,
                        width=w,
                        height=h,
                        confidence=area / (w * h),  # Use fill ratio as confidence
                        label=f"color_{color_hsv}"
                    )
                    detections.append(detection)
            
            logger.debug(f"Found {len(detections)} color matches")
            return detections
            
        except Exception as e:
            logger.error(f"Color detection failed: {e}")
            return []
    
    def find_health_bar(self, image: np.ndarray, 
                       region: Optional[Tuple[int, int, int, int]] = None) -> Optional[Detection]:
        """Find red health bar (common OSRS element)"""
        # Red color in HSV (approximate)
        red_hsv = (0, 255, 255)  # Pure red
        detections = self.find_color(image, red_hsv, tolerance=10, region=region)
        
        # Filter for health bar characteristics (horizontal rectangle)
        health_bars = []
        for det in detections:
            aspect_ratio = det.width / det.height if det.height > 0 else 0
            if 2 < aspect_ratio < 10 and det.width > 20:  # Health bars are wide and thin
                health_bars.append(det)
        
        # Return the largest health bar
        if health_bars:
            return max(health_bars, key=lambda d: d.width * d.height)
        
        return None


class FeatureDetection:
    """Feature-based object detection"""
    
    def __init__(self):
        self.algorithm = VISION["feature_detection"]["algorithm"]
        self.max_features = VISION["feature_detection"]["max_features"]
        self.match_threshold = VISION["feature_detection"]["match_threshold"]
        
        # Initialize feature detector
        if self.algorithm == "ORB":
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
        elif self.algorithm == "SIFT":
            self.detector = cv2.SIFT_create(nfeatures=self.max_features)
        else:
            logger.warning(f"Unsupported feature algorithm: {self.algorithm}")
            self.detector = cv2.ORB_create(nfeatures=self.max_features)
        
        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    def extract_features(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """Extract keypoints and descriptors from image"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            keypoints, descriptors = self.detector.detectAndCompute(gray, None)
            return keypoints, descriptors
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return [], None
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """Match features between two descriptor sets"""
        try:
            if desc1 is None or desc2 is None:
                return []
            
            matches = self.matcher.match(desc1, desc2)
            
            # Filter matches by distance
            good_matches = [m for m in matches if m.distance < self.match_threshold * 100]
            
            return good_matches
        except Exception as e:
            logger.error(f"Feature matching failed: {e}")
            return []


class ComputerVision:
    """Main computer vision class combining all detection methods"""
    
    def __init__(self):
        self.template_manager = TemplateManager()
        self.template_matching = TemplateMatching(self.template_manager)
        self.color_detection = ColorDetection()
        self.feature_detection = FeatureDetection()
        
        logger.info("Computer vision system initialized")
    
    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process image and extract useful information
        Returns dictionary with detected objects and metadata
        """
        results = {
            "timestamp": cv2.getTickCount(),
            "image_shape": image.shape,
            "detections": [],
            "health_bars": [],
            "features": {}
        }
        
        try:
            # Find health bars (common UI element)
            health_bar = self.color_detection.find_health_bar(image)
            if health_bar:
                results["health_bars"].append(health_bar)
            
            # Extract features for potential matching
            keypoints, descriptors = self.feature_detection.extract_features(image)
            results["features"] = {
                "keypoint_count": len(keypoints),
                "has_descriptors": descriptors is not None
            }
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
        
        return results
    
    def debug_visualize(self, image: np.ndarray, detections: List[Detection], 
                       window_name: str = "Debug") -> np.ndarray:
        """
        Draw detection results on image for debugging
        """
        if not DEVELOPMENT["visual_debugging"]:
            return image
        
        debug_image = image.copy()
        
        for detection in detections:
            # Draw bounding box
            cv2.rectangle(debug_image, 
                         (detection.x, detection.y),
                         (detection.x + detection.width, detection.y + detection.height),
                         (0, 255, 0), 2)
            
            # Draw confidence and label
            label = f"{detection.label}: {detection.confidence:.2f}"
            cv2.putText(debug_image, label,
                       (detection.x, detection.y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return debug_image


# Global computer vision instance
cv_system = ComputerVision() 