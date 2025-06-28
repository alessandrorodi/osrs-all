"""
YOLOv8-based Object Detection for OSRS

This module provides real-time object detection capabilities using YOLOv8
for detecting NPCs, items, players, and other game objects.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging
import torch
from ultralytics import YOLO

from core.computer_vision import Detection

logger = logging.getLogger(__name__)


@dataclass
class GameStateDetection(Detection):
    """Enhanced detection with game-specific information"""
    object_type: str = ""  # npc, item, player, ui, etc.
    game_id: Optional[int] = None  # Game object ID if known
    action_priority: float = 0.0  # Priority for AI decision making
    metadata: Optional[Dict[str, Any]] = None  # Additional object-specific data
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class YOLODetector:
    """
    YOLOv8-based object detector for OSRS game objects
    
    Detects and classifies:
    - NPCs (monsters, friendly NPCs)
    - Items (ground items, inventory items)
    - Players (other players)
    - UI elements (buttons, interfaces)
    - Environmental objects (trees, rocks, etc.)
    """
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize YOLO detector
        
        Args:
            model_path: Path to custom trained model (None for default)
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.device = self._setup_device(device)
        self.model_path = model_path
        self.model = None
        self.class_names = []
        self.confidence_threshold = 0.5
        self.iou_threshold = 0.4
        
        # Game-specific class mappings
        self.game_object_types = {
            'player': ['player', 'character'],
            'npc': ['npc', 'monster', 'enemy', 'goblin', 'cow', 'chicken'],
            'item': ['item', 'drop', 'loot', 'coin', 'rune', 'arrow'],
            'ui': ['button', 'interface', 'menu', 'inventory', 'chat'],
            'environment': ['tree', 'rock', 'ore', 'bank', 'altar', 'furnace']
        }
        
        self._load_model()
        
    def _setup_device(self, device: str) -> str:
        """Setup computation device"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info("Using CUDA for YOLO inference")
            else:
                device = "cpu"
                logger.info("Using CPU for YOLO inference")
        
        return device
    
    def _load_model(self) -> None:
        """Load YOLO model"""
        try:
            if self.model_path and Path(self.model_path).exists():
                # Load custom trained model
                self.model = YOLO(self.model_path)
                logger.info(f"Loaded custom YOLO model: {self.model_path}")
            else:
                # Use pre-trained YOLOv8 model
                self.model = YOLO('yolov8n.pt')  # nano version for speed
                logger.info("Loaded pre-trained YOLOv8n model")
            
            # Move to device
            self.model.to(self.device)
            
            # Get class names
            if hasattr(self.model, 'names'):
                self.class_names = list(self.model.names.values())
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None
    
    def detect_objects(self, image: np.ndarray, 
                      confidence: Optional[float] = None,
                      classes: Optional[List[str]] = None) -> List[GameStateDetection]:
        """
        Detect objects in image using YOLO
        
        Args:
            image: Input image (BGR format)
            confidence: Confidence threshold (overrides default)
            classes: Specific classes to detect (None for all)
            
        Returns:
            List of detected objects as GameStateDetection objects
        """
        if self.model is None:
            logger.warning("YOLO model not loaded")
            return []
        
        try:
            # Set confidence threshold
            conf_threshold = confidence or self.confidence_threshold
            
            # Run inference
            results = self.model(
                image,
                conf=conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for i in range(len(boxes)):
                        # Extract box coordinates
                        xyxy = boxes.xyxy[i].cpu().numpy()
                        x1, y1, x2, y2 = xyxy.astype(int)
                        
                        # Extract confidence and class
                        conf = float(boxes.conf[i].cpu().numpy())
                        cls_id = int(boxes.cls[i].cpu().numpy())
                        
                        # Get class name
                        class_name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                        
                        # Filter by classes if specified
                        if classes and class_name not in classes:
                            continue
                        
                        # Determine object type
                        object_type = self._classify_game_object(class_name)
                        
                        # Calculate action priority
                        priority = self._calculate_priority(object_type, class_name, conf)
                        
                        # Create detection
                        detection = GameStateDetection(
                            x=x1,
                            y=y1,
                            width=x2 - x1,
                            height=y2 - y1,
                            confidence=conf,
                            label=class_name,
                            object_type=object_type,
                            action_priority=priority,
                            metadata={
                                'class_id': cls_id,
                                'detection_method': 'yolo'
                            }
                        )
                        
                        detections.append(detection)
            
            logger.debug(f"YOLOv8 detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    def _classify_game_object(self, class_name: str) -> str:
        """Classify detected object into game categories"""
        class_name_lower = class_name.lower()
        
        for object_type, keywords in self.game_object_types.items():
            if any(keyword in class_name_lower for keyword in keywords):
                return object_type
        
        return "unknown"
    
    def _calculate_priority(self, object_type: str, class_name: str, confidence: float) -> float:
        """Calculate action priority for AI decision making"""
        base_priorities = {
            'player': 0.3,  # Medium priority - social aspects
            'npc': 0.8,     # High priority - combat/interaction targets
            'item': 0.9,    # Very high priority - loot
            'ui': 0.6,      # Medium-high priority - interface interactions
            'environment': 0.5,  # Medium priority - skilling objects
            'unknown': 0.1  # Low priority
        }
        
        base_priority = base_priorities.get(object_type, 0.1)
        
        # Adjust based on confidence
        confidence_multiplier = min(confidence * 1.2, 1.0)
        
        # Special adjustments for specific objects
        priority_adjustments = {
            'coin': 1.0,    # Always high priority
            'rare': 1.0,    # High value items
            'enemy': 0.9,   # Combat targets
            'bank': 0.8,    # Important utility
        }
        
        for keyword, multiplier in priority_adjustments.items():
            if keyword in class_name.lower():
                confidence_multiplier *= multiplier
                break
        
        return min(base_priority * confidence_multiplier, 1.0)
    
    def detect_specific_objects(self, image: np.ndarray, object_types: List[str],
                              confidence: float = 0.5) -> Dict[str, List[GameStateDetection]]:
        """
        Detect specific types of objects
        
        Args:
            image: Input image
            object_types: List of object types to detect ('npc', 'item', etc.)
            confidence: Confidence threshold
            
        Returns:
            Dictionary mapping object types to their detections
        """
        all_detections = self.detect_objects(image, confidence=confidence)
        
        results = {obj_type: [] for obj_type in object_types}
        
        for detection in all_detections:
            if detection.object_type in object_types:
                results[detection.object_type].append(detection)
        
        return results
    
    def get_highest_priority_objects(self, image: np.ndarray, 
                                   top_k: int = 5) -> List[GameStateDetection]:
        """
        Get the highest priority objects for AI decision making
        
        Args:
            image: Input image
            top_k: Number of top priority objects to return
            
        Returns:
            List of highest priority detections
        """
        detections = self.detect_objects(image)
        
        # Sort by priority (descending)
        detections.sort(key=lambda d: d.action_priority, reverse=True)
        
        return detections[:top_k]
    
    def update_model(self, model_path: str) -> bool:
        """
        Update the YOLO model with a new trained model
        
        Args:
            model_path: Path to new model file
            
        Returns:
            True if model loaded successfully
        """
        try:
            if not Path(model_path).exists():
                logger.error(f"Model file not found: {model_path}")
                return False
            
            self.model_path = model_path
            self._load_model()
            
            if self.model is not None:
                logger.info(f"Successfully updated YOLO model: {model_path}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to update YOLO model: {e}")
            return False
    
    def visualize_detections(self, image: np.ndarray, 
                           detections: List[GameStateDetection],
                           show_priority: bool = True) -> np.ndarray:
        """
        Visualize detections on image for debugging
        
        Args:
            image: Input image
            detections: List of detections
            show_priority: Whether to show priority scores
            
        Returns:
            Image with visualized detections
        """
        vis_image = image.copy()
        
        # Color mapping for different object types
        colors = {
            'player': (255, 0, 0),      # Blue
            'npc': (0, 255, 0),         # Green
            'item': (0, 0, 255),        # Red
            'ui': (255, 255, 0),        # Cyan
            'environment': (255, 0, 255), # Magenta
            'unknown': (128, 128, 128)   # Gray
        }
        
        for detection in detections:
            color = colors.get(detection.object_type, colors['unknown'])
            
            # Draw bounding box
            cv2.rectangle(vis_image, 
                         (detection.x, detection.y),
                         (detection.x + detection.width, detection.y + detection.height),
                         color, 2)
            
            # Prepare label
            label = f"{detection.label} ({detection.confidence:.2f})"
            if show_priority:
                label += f" P:{detection.action_priority:.2f}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(vis_image,
                         (detection.x, detection.y - label_size[1] - 10),
                         (detection.x + label_size[0], detection.y),
                         color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label,
                       (detection.x, detection.y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return vis_image