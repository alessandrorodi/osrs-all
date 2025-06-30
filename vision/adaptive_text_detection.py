#!/usr/bin/env python3
"""
Adaptive Text Detection for OSRS
No hardcoded regions - analyzes full screenshot and detects UI elements dynamically
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)

@dataclass
class DetectedTextArea:
    """Dynamically detected text area"""
    name: str
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    text_type: str  # chat, interface, numbers, items
    confidence: float
    texts: List[str]  # All text found in this area

class AdaptiveTextDetector:
    """
    Adaptive text detection that works with any OSRS client layout
    - No hardcoded regions
    - Works with RuneLite, vanilla OSRS, any resolution
    - Detects UI elements first, then applies OCR
    """
    
    def __init__(self):
        from vision.detectors.ocr_detector import OCRDetector
        self.ocr = OCRDetector()
        
        # UI element detection patterns
        self.ui_colors = {
            'chat_background': [(0, 0, 0), (50, 50, 50)],      # Dark backgrounds
            'interface_brown': [(101, 68, 52), (150, 120, 80)], # OSRS brown interface
            'health_red': [(150, 0, 0), (255, 50, 50)],         # Health orb red
            'prayer_blue': [(0, 100, 150), (50, 150, 255)],     # Prayer orb blue
            'energy_yellow': [(200, 200, 0), (255, 255, 100)]   # Energy yellow
        }
    
    def detect_all_text(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Detect all text in screenshot without predefined regions
        
        Args:
            screenshot: Full OSRS screenshot
            
        Returns:
            Dictionary with all detected text organized by type
        """
        start_time = time.time()
        
        try:
            logger.info(f"Analyzing full screenshot: {screenshot.shape[1]}x{screenshot.shape[0]}")
            
            # Step 1: Run OCR on entire screenshot
            all_text_detections = self.ocr.detect_text(screenshot, confidence_threshold=0.3)
            logger.info(f"Found {len(all_text_detections)} text detections")
            
            # Step 2: Group text by location and type
            text_areas = self._group_text_by_areas(all_text_detections, screenshot)
            
            # Step 3: Classify text areas by content and location
            classified_areas = self._classify_text_areas(text_areas, screenshot)
            
            # Step 4: Extract structured data
            results = self._extract_structured_data(classified_areas)
            
            # Step 5: Add performance metrics
            processing_time = time.time() - start_time
            results['performance'] = {
                'processing_time': processing_time,
                'total_detections': len(all_text_detections),
                'text_areas': len(text_areas)
            }
            
            logger.info(f"Adaptive text detection completed in {processing_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Adaptive text detection failed: {e}")
            return {
                'chat_messages': [],
                'items': [],
                'player_stats': {},
                'interface_elements': [],
                'error': str(e)
            }
    
    def _group_text_by_areas(self, detections: List, screenshot: np.ndarray) -> List[DetectedTextArea]:
        """Group nearby text detections into logical areas"""
        if not detections:
            return []
        
        # Sort detections by position
        detections.sort(key=lambda d: (d.y, d.x))
        
        areas = []
        current_area_texts = []
        current_area_bbox = None
        
        for detection in detections:
            # If this is the first detection or it's far from current area
            if (current_area_bbox is None or 
                self._is_far_from_area(detection, current_area_bbox)):
                
                # Save previous area if it exists
                if current_area_texts:
                    area = self._create_text_area(current_area_texts, current_area_bbox)
                    if area:
                        areas.append(area)
                
                # Start new area
                current_area_texts = [detection]
                current_area_bbox = [detection.x, detection.y, 
                                   detection.x + detection.width, 
                                   detection.y + detection.height]
            else:
                # Add to current area
                current_area_texts.append(detection)
                # Expand bounding box
                current_area_bbox[0] = min(current_area_bbox[0], detection.x)
                current_area_bbox[1] = min(current_area_bbox[1], detection.y)
                current_area_bbox[2] = max(current_area_bbox[2], detection.x + detection.width)
                current_area_bbox[3] = max(current_area_bbox[3], detection.y + detection.height)
        
        # Don't forget the last area
        if current_area_texts:
            area = self._create_text_area(current_area_texts, current_area_bbox)
            if area:
                areas.append(area)
        
        return areas
    
    def _is_far_from_area(self, detection, area_bbox: List[int], threshold: int = 100) -> bool:
        """Check if detection is far from the current area"""
        det_center_x = detection.x + detection.width // 2
        det_center_y = detection.y + detection.height // 2
        
        area_center_x = (area_bbox[0] + area_bbox[2]) // 2
        area_center_y = (area_bbox[1] + area_bbox[3]) // 2
        
        distance = ((det_center_x - area_center_x) ** 2 + 
                   (det_center_y - area_center_y) ** 2) ** 0.5
        
        return distance > threshold
    
    def _create_text_area(self, texts: List, bbox: List[int]) -> Optional[DetectedTextArea]:
        """Create a text area from grouped detections"""
        if not texts:
            return None
        
        # Extract all text content
        text_content = [t.text for t in texts]
        
        # Determine area type based on content
        area_type = self._classify_text_content(text_content)
        
        # Calculate confidence (average of all detections)
        confidence = sum(t.confidence for t in texts) / len(texts)
        
        return DetectedTextArea(
            name=f"{area_type}_area_{len(text_content)}",
            bbox=(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]),
            text_type=area_type,
            confidence=confidence,
            texts=text_content
        )
    
    def _classify_text_content(self, texts: List[str]) -> str:
        """Classify what type of text this area contains"""
        combined_text = ' '.join(texts).lower()
        
        # Chat indicators
        if any(indicator in combined_text for indicator in [':', 'says', 'yells', 'whispers', 'traded']):
            return 'chat'
        
        # Numbers/stats
        if any(char.isdigit() for text in texts for char in text) and len(combined_text) < 10:
            return 'numbers'
        
        # Interface elements
        if any(word in combined_text for word in ['click', 'attack', 'examine', 'use', 'talk', 'bank', 'trade']):
            return 'interface'
        
        # Items (typically short text)
        if all(len(text) < 20 for text in texts) and not any(char.isdigit() for char in combined_text):
            return 'items'
        
        # Default to interface
        return 'interface'
    
    def _classify_text_areas(self, areas: List[DetectedTextArea], screenshot: np.ndarray) -> List[DetectedTextArea]:
        """Further classify text areas based on screen position and visual context"""
        h, w = screenshot.shape[:2]
        
        for area in areas:
            x, y, width, height = area.bbox
            
            # Refine classification based on position
            
            # Bottom area is likely chat
            if y > h * 0.8:  # Bottom 20% of screen
                if area.text_type in ['interface', 'chat']:
                    area.text_type = 'chat'
                    area.name = f"chat_bottom_{x}"
            
            # Right side is likely inventory/interface
            elif x > w * 0.7:  # Right 30% of screen
                if area.text_type in ['items', 'interface']:
                    area.text_type = 'inventory'
                    area.name = f"inventory_{y}"
            
            # Top area might be overlays/stats
            elif y < h * 0.2:  # Top 20% of screen
                area.text_type = 'overlay'
                area.name = f"overlay_{x}"
            
            # Center areas are likely interface
            else:
                if area.text_type == 'interface':
                    area.name = f"interface_{x}_{y}"
        
        return areas
    
    def _extract_structured_data(self, areas: List[DetectedTextArea]) -> Dict[str, Any]:
        """Extract structured data from classified text areas"""
        results = {
            'chat_messages': [],
            'items': [],
            'player_stats': {},
            'interface_elements': [],
            'overlays': [],
            'numbers': []
        }
        
        for area in areas:
            if area.text_type == 'chat':
                # Parse chat messages
                for text in area.texts:
                    if ':' in text and len(text) > 3:
                        results['chat_messages'].append({
                            'text': text,
                            'position': area.bbox,
                            'confidence': area.confidence
                        })
            
            elif area.text_type == 'inventory':
                # Parse items
                for text in area.texts:
                    if len(text) > 2:
                        results['items'].append({
                            'name': text,
                            'position': area.bbox,
                            'confidence': area.confidence
                        })
            
            elif area.text_type == 'numbers':
                # Parse numeric values
                for text in area.texts:
                    if any(char.isdigit() for char in text):
                        results['numbers'].append({
                            'value': text,
                            'position': area.bbox,
                            'confidence': area.confidence
                        })
            
            elif area.text_type == 'overlay':
                # RuneLite overlays, stats, etc.
                results['overlays'].append({
                    'texts': area.texts,
                    'position': area.bbox,
                    'confidence': area.confidence
                })
            
            else:  # interface
                results['interface_elements'].extend([{
                    'text': text,
                    'position': area.bbox,
                    'confidence': area.confidence
                } for text in area.texts])
        
        return results
    
    def get_summary(self, results: Dict[str, Any]) -> Dict[str, int]:
        """Get a summary of detected text"""
        return {
            'chat_messages': len(results.get('chat_messages', [])),
            'items': len(results.get('items', [])),
            'interface_elements': len(results.get('interface_elements', [])),
            'overlays': len(results.get('overlays', [])),
            'numbers': len(results.get('numbers', [])),
            'processing_time': results.get('performance', {}).get('processing_time', 0)
        }

# Global instance
adaptive_text_detector = AdaptiveTextDetector() 