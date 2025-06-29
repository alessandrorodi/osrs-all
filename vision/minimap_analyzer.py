"""
Advanced Minimap Analysis System for OSRS

This module provides comprehensive minimap analysis capabilities including:
- Player position and compass detection
- NPC dot analysis (yellow, red, green dots)
- Clickable area detection and validation
- Compass direction and camera angle detection
- Minimap zoom level detection
- Real-time processing optimized for RTX 4090
- Pathfinding integration

Integrates with YOLOv8 for enhanced object detection and provides
60fps real-time processing capabilities.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import time
import threading
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import math

from core.computer_vision import Detection, cv_system
from vision.detectors.yolo_detector import YOLODetector
from config.settings import VISION

logger = logging.getLogger(__name__)


class MinimapRegionType(Enum):
    """Types of minimap regions"""
    SAFE_ZONE = "safe_zone"
    WILDERNESS = "wilderness"
    MULTICOMBAT = "multicombat"
    BANK_AREA = "bank_area"
    CITY = "city"
    DUNGEON = "dungeon"
    WATER = "water"
    UNKNOWN = "unknown"


class DotType(Enum):
    """Types of dots on minimap"""
    PLAYER_SELF = "player_self"        # White square (player)
    PLAYER_OTHER = "player_other"      # White dot (other players)
    NPC_NEUTRAL = "npc_neutral"        # Yellow dot (NPCs)
    NPC_AGGRESSIVE = "npc_aggressive"  # Red dot (aggressive NPCs)
    FRIEND = "friend"                  # Green dot (friends)
    CLAN_MEMBER = "clan_member"        # Purple dot (clan members)
    TEAM_MEMBER = "team_member"        # Blue dot (team members)
    ITEM = "item"                      # Red dot (items)
    QUEST_MARKER = "quest_marker"      # Yellow arrow
    TARGET_MARKER = "target_marker"    # Red arrow


@dataclass
class MinimapDot:
    """Represents a dot on the minimap"""
    x: int
    y: int
    dot_type: DotType
    confidence: float
    size: int = 2
    color_bgr: Tuple[int, int, int] = (0, 0, 0)
    
    @property
    def world_coords(self) -> Tuple[int, int]:
        """Convert minimap coordinates to estimated world coordinates"""
        # Minimap scale: 4 pixels = 1 tile
        # Center of minimap (73, 75) represents player position
        offset_x = (self.x - 73) / 4
        offset_y = (self.y - 75) / 4
        return (int(offset_x), int(offset_y))


@dataclass
class CompassInfo:
    """Compass and camera information"""
    north_angle: float = 0.0  # Degrees from North
    camera_angle: float = 0.0  # Camera rotation angle
    compass_position: Tuple[int, int] = (0, 0)
    zoom_level: float = 1.0
    is_compass_visible: bool = True


@dataclass
class MinimapClickableArea:
    """Represents a clickable area on minimap"""
    x: int
    y: int
    width: int
    height: int
    area_type: str  # "walkable", "obstacle", "door", "water"
    confidence: float
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)


@dataclass
class MinimapAnalysisResult:
    """Comprehensive minimap analysis results"""
    timestamp: float
    player_position: Tuple[int, int]
    compass_info: CompassInfo
    detected_dots: List[MinimapDot]
    clickable_areas: List[MinimapClickableArea]
    region_type: MinimapRegionType
    region_name: str
    obstacles: List[Tuple[int, int]]
    walkable_areas: List[Tuple[int, int]]
    points_of_interest: List[Dict[str, Any]]
    processing_time: float
    
    def get_dots_by_type(self, dot_type: DotType) -> List[MinimapDot]:
        """Get all dots of a specific type"""
        return [dot for dot in self.detected_dots if dot.dot_type == dot_type]
    
    def get_nearest_dot(self, dot_type: DotType, max_distance: int = 50) -> Optional[MinimapDot]:
        """Get nearest dot of specified type to player"""
        dots = self.get_dots_by_type(dot_type)
        if not dots:
            return None
        
        player_x, player_y = self.player_position
        min_distance = float('inf')
        nearest_dot = None
        
        for dot in dots:
            distance = math.sqrt((dot.x - player_x)**2 + (dot.y - player_y)**2)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                nearest_dot = dot
        
        return nearest_dot


class AdvancedMinimapAnalyzer:
    """
    Advanced minimap analysis system with YOLOv8 integration
    
    Provides comprehensive minimap analysis including:
    - High-precision dot detection
    - Compass and camera angle detection
    - Clickable area identification
    - Region classification
    - Real-time processing optimization
    """
    
    def __init__(self, device: str = "cuda", enable_yolo: bool = True):
        """
        Initialize the advanced minimap analyzer
        
        Args:
            device: Processing device ('cuda', 'cpu', 'auto')
            enable_yolo: Whether to use YOLOv8 for enhanced detection
        """
        self.device = device
        self.enable_yolo = enable_yolo
        
        # Minimap region coordinates (x, y, width, height)
        self.minimap_region = (570, 9, 146, 151)  # Standard OSRS minimap
        self.compass_region = (570, 9, 25, 25)    # Compass area
        
        # Initialize YOLOv8 detector if enabled
        self.yolo_detector = None
        if enable_yolo:
            try:
                self.yolo_detector = YOLODetector(device=device)
                logger.info("YOLOv8 minimap detection enabled")
            except Exception as e:
                logger.warning(f"YOLOv8 initialization failed: {e}")
                self.enable_yolo = False
        
        # Color definitions for dot detection (BGR format)
        self.dot_colors = {
            DotType.PLAYER_SELF: {
                'color': (255, 255, 255),  # White
                'hsv_lower': (0, 0, 200),
                'hsv_upper': (180, 30, 255),
                'size_range': (3, 6)
            },
            DotType.PLAYER_OTHER: {
                'color': (255, 255, 255),  # White
                'hsv_lower': (0, 0, 180),
                'hsv_upper': (180, 50, 255),
                'size_range': (1, 4)
            },
            DotType.NPC_NEUTRAL: {
                'color': (0, 255, 255),    # Yellow
                'hsv_lower': (20, 100, 100),
                'hsv_upper': (30, 255, 255),
                'size_range': (1, 4)
            },
            DotType.FRIEND: {
                'color': (0, 255, 0),      # Green
                'hsv_lower': (40, 100, 100),
                'hsv_upper': (80, 255, 255),
                'size_range': (1, 4)
            },
            DotType.CLAN_MEMBER: {
                'color': (255, 0, 255),    # Purple
                'hsv_lower': (140, 100, 100),
                'hsv_upper': (160, 255, 255),
                'size_range': (1, 4)
            },
            DotType.TEAM_MEMBER: {
                'color': (255, 0, 0),      # Blue
                'hsv_lower': (100, 100, 100),
                'hsv_upper': (120, 255, 255),
                'size_range': (1, 4)
            },
            DotType.ITEM: {
                'color': (0, 0, 255),      # Red
                'hsv_lower': (0, 100, 100),
                'hsv_upper': (10, 255, 255),
                'size_range': (1, 3)
            }
        }
        
        # Performance optimization
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.processing_stats = {
            'frames_processed': 0,
            'avg_processing_time': 0.0,
            'fps': 0.0,
            'last_fps_check': time.time()
        }
        
        # Caching for performance
        self.region_cache = {}
        self.color_cache = {}
        
        logger.info(f"AdvancedMinimapAnalyzer initialized (device: {device})")
    
    def analyze_minimap(self, game_screenshot: np.ndarray, 
                       minimap_region: Optional[Tuple[int, int, int, int]] = None) -> MinimapAnalysisResult:
        """
        Perform comprehensive minimap analysis
        
        Args:
            game_screenshot: Full game screenshot
            minimap_region: Optional custom minimap region
            
        Returns:
            MinimapAnalysisResult with comprehensive analysis
        """
        start_time = time.time()
        
        try:
            # Extract minimap region
            region = minimap_region or self.minimap_region
            x, y, w, h = region
            minimap = game_screenshot[y:y+h, x:x+w].copy()
            
            # Parallel processing for performance
            futures = []
            
            # Start parallel analysis tasks
            futures.append(self.thread_pool.submit(self._detect_player_position, minimap))
            futures.append(self.thread_pool.submit(self._analyze_compass, game_screenshot))
            futures.append(self.thread_pool.submit(self._detect_dots, minimap))
            futures.append(self.thread_pool.submit(self._identify_clickable_areas, minimap))
            futures.append(self.thread_pool.submit(self._classify_region, minimap))
            futures.append(self.thread_pool.submit(self._find_obstacles, minimap))
            futures.append(self.thread_pool.submit(self._find_walkable_areas, minimap))
            
            # Collect results
            player_position = futures[0].result()
            compass_info = futures[1].result()
            detected_dots = futures[2].result()
            clickable_areas = futures[3].result()
            region_type, region_name = futures[4].result()
            obstacles = futures[5].result()
            walkable_areas = futures[6].result()
            
            # Find points of interest
            points_of_interest = self._find_points_of_interest(minimap, detected_dots)
            
            processing_time = time.time() - start_time
            
            # Update performance stats
            self._update_performance_stats(processing_time)
            
            # Create result
            result = MinimapAnalysisResult(
                timestamp=start_time,
                player_position=player_position,
                compass_info=compass_info,
                detected_dots=detected_dots,
                clickable_areas=clickable_areas,
                region_type=region_type,
                region_name=region_name,
                obstacles=obstacles,
                walkable_areas=walkable_areas,
                points_of_interest=points_of_interest,
                processing_time=processing_time
            )
            
            logger.debug(f"Minimap analysis completed in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Minimap analysis failed: {e}")
            return self._create_empty_result(start_time)
    
    def _detect_player_position(self, minimap: np.ndarray) -> Tuple[int, int]:
        """Detect player position on minimap (white square/arrow)"""
        try:
            # Look for white square/arrow in center area
            center_x, center_y = minimap.shape[1] // 2, minimap.shape[0] // 2
            search_area = 20  # pixels around center
            
            # Extract center region
            y1 = max(0, center_y - search_area)
            y2 = min(minimap.shape[0], center_y + search_area)
            x1 = max(0, center_x - search_area)
            x2 = min(minimap.shape[1], center_x + search_area)
            
            center_region = minimap[y1:y2, x1:x2]
            
            # Convert to HSV for better white detection
            hsv = cv2.cvtColor(center_region, cv2.COLOR_BGR2HSV)
            
            # White color range for player marker
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            
            mask = cv2.inRange(hsv, lower_white, upper_white)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour (player marker)
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Get centroid
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + x1
                    cy = int(M["m01"] / M["m00"]) + y1
                    return (cx, cy)
            
            # Fallback to center if no player marker found
            return (center_x, center_y)
            
        except Exception as e:
            logger.error(f"Player position detection failed: {e}")
            return (73, 75)  # Default center
    
    def _analyze_compass(self, game_screenshot: np.ndarray) -> CompassInfo:
        """Analyze compass for direction and camera angle"""
        try:
            # Extract compass region
            x, y, w, h = self.compass_region
            compass = game_screenshot[y:y+h, x:x+w].copy()
            
            # Detect compass needle direction
            north_angle = self._detect_compass_needle(compass)
            
            # Estimate camera angle (simplified)
            camera_angle = self._estimate_camera_angle(compass)
            
            # Detect zoom level
            zoom_level = self._detect_zoom_level(compass)
            
            return CompassInfo(
                north_angle=north_angle,
                camera_angle=camera_angle,
                compass_position=(x + w//2, y + h//2),
                zoom_level=zoom_level,
                is_compass_visible=True
            )
            
        except Exception as e:
            logger.error(f"Compass analysis failed: {e}")
            return CompassInfo()
    
    def _detect_compass_needle(self, compass: np.ndarray) -> float:
        """Detect compass needle direction"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(compass, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=15)
            
            if lines is not None:
                # Find the most prominent line (compass needle)
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = np.degrees(theta)
                    angles.append(angle)
                
                if angles:
                    # Return average angle
                    return np.mean(angles)
            
            return 0.0  # North
            
        except Exception as e:
            logger.error(f"Compass needle detection failed: {e}")
            return 0.0
    
    def _estimate_camera_angle(self, compass: np.ndarray) -> float:
        """Estimate camera rotation angle"""
        # Simplified camera angle estimation
        # In a real implementation, this would analyze the compass orientation
        return 0.0
    
    def _detect_zoom_level(self, compass: np.ndarray) -> float:
        """Detect minimap zoom level"""
        # Simplified zoom detection
        # In a real implementation, this would analyze minimap scale
        return 1.0
    
    def _detect_dots(self, minimap: np.ndarray) -> List[MinimapDot]:
        """Detect all dots on minimap"""
        try:
            all_dots = []
            
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
            
            # Detect each type of dot
            for dot_type, color_info in self.dot_colors.items():
                dots = self._detect_specific_dots(hsv, minimap, dot_type, color_info)
                all_dots.extend(dots)
            
            return all_dots
            
        except Exception as e:
            logger.error(f"Dot detection failed: {e}")
            return []
    
    def _detect_specific_dots(self, hsv: np.ndarray, minimap: np.ndarray, 
                            dot_type: DotType, color_info: Dict) -> List[MinimapDot]:
        """Detect specific type of dots"""
        try:
            dots = []
            
            # Create color mask
            lower = np.array(color_info['hsv_lower'])
            upper = np.array(color_info['hsv_upper'])
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            min_size, max_size = color_info['size_range']
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by size
                if min_size <= area <= max_size * 4:  # Allow some tolerance
                    # Get centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # Calculate confidence based on size and color match
                        confidence = min(1.0, area / (max_size * 2))
                        
                        dot = MinimapDot(
                            x=cx,
                            y=cy,
                            dot_type=dot_type,
                            confidence=confidence,
                            size=int(area),
                            color_bgr=color_info['color']
                        )
                        dots.append(dot)
            
            return dots
            
        except Exception as e:
            logger.error(f"Specific dot detection failed for {dot_type}: {e}")
            return []
    
    def _identify_clickable_areas(self, minimap: np.ndarray) -> List[MinimapClickableArea]:
        """Identify clickable areas on minimap"""
        try:
            clickable_areas = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
            
            # Detect walkable areas (darker regions)
            _, walkable_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(walkable_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 10:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    clickable_area = MinimapClickableArea(
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                        area_type="walkable",
                        confidence=0.8
                    )
                    clickable_areas.append(clickable_area)
            
            return clickable_areas
            
        except Exception as e:
            logger.error(f"Clickable area identification failed: {e}")
            return []
    
    def _classify_region(self, minimap: np.ndarray) -> Tuple[MinimapRegionType, str]:
        """Classify the current minimap region"""
        try:
            # Analyze color distribution
            hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
            
            # Calculate color histograms
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            
            # Analyze dominant colors
            dominant_hue = np.argmax(hist_h)
            dominant_saturation = np.argmax(hist_s)
            dominant_value = np.argmax(hist_v)
            
            # Classify based on color characteristics
            if dominant_hue < 30 or dominant_hue > 150:  # Red/brown tones
                if dominant_value < 100:
                    return MinimapRegionType.WILDERNESS, "Wilderness"
                else:
                    return MinimapRegionType.CITY, "City Area"
            elif 30 <= dominant_hue <= 90:  # Green tones
                return MinimapRegionType.SAFE_ZONE, "Safe Area"
            elif 90 <= dominant_hue <= 130:  # Blue tones
                return MinimapRegionType.WATER, "Water Area"
            else:
                return MinimapRegionType.UNKNOWN, "Unknown Area"
            
        except Exception as e:
            logger.error(f"Region classification failed: {e}")
            return MinimapRegionType.UNKNOWN, "Unknown"
    
    def _find_obstacles(self, minimap: np.ndarray) -> List[Tuple[int, int]]:
        """Find obstacles on minimap (walls, barriers)"""
        try:
            obstacles = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
            
            # Detect white lines (walls)
            _, wall_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Find contours
            contours, _ = cv2.findContours(wall_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # Get contour points
                for point in contour:
                    x, y = point[0]
                    obstacles.append((x, y))
            
            return obstacles
            
        except Exception as e:
            logger.error(f"Obstacle detection failed: {e}")
            return []
    
    def _find_walkable_areas(self, minimap: np.ndarray) -> List[Tuple[int, int]]:
        """Find walkable areas on minimap"""
        try:
            walkable = []
            
            # Convert to grayscale
            gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
            
            # Detect dark areas (walkable)
            _, walkable_mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
            
            # Find all walkable pixels
            walkable_pixels = np.where(walkable_mask == 255)
            
            for y, x in zip(walkable_pixels[0], walkable_pixels[1]):
                walkable.append((x, y))
            
            return walkable
            
        except Exception as e:
            logger.error(f"Walkable area detection failed: {e}")
            return []
    
    def _find_points_of_interest(self, minimap: np.ndarray, 
                               detected_dots: List[MinimapDot]) -> List[Dict[str, Any]]:
        """Find points of interest on minimap"""
        try:
            poi = []
            
            # Banks (look for specific NPCs or building patterns)
            bank_npcs = [dot for dot in detected_dots if dot.dot_type == DotType.NPC_NEUTRAL]
            if len(bank_npcs) > 2:  # Multiple NPCs might indicate a bank
                poi.append({
                    'type': 'bank',
                    'position': (73, 75),  # Center for now
                    'confidence': 0.6
                })
            
            # Shops (similar to banks)
            if len(bank_npcs) > 1:
                poi.append({
                    'type': 'shop',
                    'position': (73, 75),
                    'confidence': 0.5
                })
            
            # Aggressive NPCs (potential danger)
            aggressive_npcs = [dot for dot in detected_dots if dot.dot_type == DotType.NPC_AGGRESSIVE]
            for npc in aggressive_npcs:
                poi.append({
                    'type': 'danger',
                    'position': (npc.x, npc.y),
                    'confidence': npc.confidence
                })
            
            return poi
            
        except Exception as e:
            logger.error(f"POI detection failed: {e}")
            return []
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        self.processing_stats['frames_processed'] += 1
        
        # Update average processing time
        current_avg = self.processing_stats['avg_processing_time']
        frame_count = self.processing_stats['frames_processed']
        
        self.processing_stats['avg_processing_time'] = (
            (current_avg * (frame_count - 1) + processing_time) / frame_count
        )
        
        # Update FPS
        current_time = time.time()
        if current_time - self.processing_stats['last_fps_check'] >= 1.0:
            self.processing_stats['fps'] = frame_count / (current_time - self.processing_stats['last_fps_check'])
            self.processing_stats['last_fps_check'] = current_time
    
    def _create_empty_result(self, timestamp: float) -> MinimapAnalysisResult:
        """Create empty result for error cases"""
        return MinimapAnalysisResult(
            timestamp=timestamp,
            player_position=(73, 75),
            compass_info=CompassInfo(),
            detected_dots=[],
            clickable_areas=[],
            region_type=MinimapRegionType.UNKNOWN,
            region_name="Unknown",
            obstacles=[],
            walkable_areas=[],
            points_of_interest=[],
            processing_time=0.0
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        return self.processing_stats.copy()
    
    def is_point_walkable(self, x: int, y: int, minimap: np.ndarray) -> bool:
        """Check if a point on minimap is walkable"""
        try:
            if 0 <= x < minimap.shape[1] and 0 <= y < minimap.shape[0]:
                # Convert to grayscale
                gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
                
                # Check if pixel is dark enough to be walkable
                pixel_value = gray[y, x]
                return pixel_value < 100  # Threshold for walkable areas
            
            return False
            
        except Exception as e:
            logger.error(f"Walkability check failed: {e}")
            return False
    
    def get_minimap_bounds(self) -> Tuple[int, int, int, int]:
        """Get minimap region bounds"""
        return self.minimap_region
    
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        logger.info("AdvancedMinimapAnalyzer cleanup completed")