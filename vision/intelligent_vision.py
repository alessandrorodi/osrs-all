"""
Intelligent Vision System for OSRS - Phase 2 Implementation

This module implements the comprehensive AI vision system that combines
YOLOv8, OCR, minimap analysis, and scene classification for advanced
game state understanding as specified in the ULTIMATE_AI_VISION plan.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import time
from enum import Enum

from core.computer_vision import Detection, cv_system
from vision.detectors.yolo_detector import YOLODetector, GameStateDetection
from vision.detectors.ocr_detector import OCRDetector, TextDetection

logger = logging.getLogger(__name__)


class SceneType(Enum):
    """Types of game scenes/environments"""
    COMBAT = "combat"
    SKILLING = "skilling"
    BANKING = "banking"
    TRADING = "trading"
    QUESTING = "questing"
    WILDERNESS = "wilderness"
    MINIGAME = "minigame"
    LOBBY = "lobby"
    UNKNOWN = "unknown"


@dataclass
class PlayerStatus:
    """Player's current status"""
    health_percent: float = 100.0
    prayer_percent: float = 100.0
    energy_percent: float = 100.0
    combat_level: Optional[int] = None
    is_in_combat: bool = False
    is_moving: bool = False
    animation_state: str = "idle"


@dataclass
class MinimapInfo:
    """Minimap analysis results"""
    player_position: Optional[Tuple[int, int]] = None
    north_direction: float = 0.0  # degrees
    visible_npcs: Optional[List[Tuple[int, int]]] = None
    visible_players: Optional[List[Tuple[int, int]]] = None
    points_of_interest: Optional[List[Dict[str, Any]]] = None
    region_type: str = "unknown"
    
    def __post_init__(self):
        if self.visible_npcs is None:
            self.visible_npcs = []
        if self.visible_players is None:
            self.visible_players = []
        if self.points_of_interest is None:
            self.points_of_interest = []


@dataclass
class InventoryInfo:
    """Inventory analysis results"""
    items: Optional[List[Dict[str, Any]]] = None
    free_slots: int = 28
    valuable_items: Optional[List[str]] = None
    consumables: Optional[List[str]] = None
    equipment: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.items is None:
            self.items = []
        if self.valuable_items is None:
            self.valuable_items = []
        if self.consumables is None:
            self.consumables = []
        if self.equipment is None:
            self.equipment = []


@dataclass
class InterfaceState:
    """Current interface state"""
    open_interfaces: Optional[List[str]] = None
    clickable_elements: Optional[List[Dict[str, Any]]] = None
    dialog_text: Optional[List[str]] = None
    active_chat: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.open_interfaces is None:
            self.open_interfaces = []
        if self.clickable_elements is None:
            self.clickable_elements = []
        if self.dialog_text is None:
            self.dialog_text = []
        if self.active_chat is None:
            self.active_chat = []


@dataclass
class GameState:
    """Comprehensive game state representation"""
    timestamp: float
    scene_type: SceneType
    confidence: float
    
    # Core components
    player_status: PlayerStatus
    minimap: MinimapInfo
    inventory: InventoryInfo
    interface_state: InterfaceState
    
    # Detected objects
    npcs: Optional[List[GameStateDetection]] = None
    items: Optional[List[GameStateDetection]] = None
    players: Optional[List[GameStateDetection]] = None
    ui_elements: Optional[List[GameStateDetection]] = None
    environment: Optional[List[GameStateDetection]] = None
    
    # Analysis metadata
    processing_time: float = 0.0
    frame_quality: float = 1.0
    analysis_version: str = "2.0"
    
    def __post_init__(self):
        if self.npcs is None:
            self.npcs = []
        if self.items is None:
            self.items = []
        if self.players is None:
            self.players = []
        if self.ui_elements is None:
            self.ui_elements = []
        if self.environment is None:
            self.environment = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    def get_highest_priority_objects(self, top_k: int = 5) -> List[GameStateDetection]:
        """Get the highest priority objects for AI decision making"""
        all_objects = self.npcs + self.items + self.players + self.ui_elements + self.environment
        all_objects.sort(key=lambda obj: obj.action_priority, reverse=True)
        return all_objects[:top_k]


class MinimapAnalyzer:
    """Minimap analysis and intelligence"""
    
    def __init__(self):
        self.minimap_region = (570, 9, 146, 151)  # Typical minimap location
        self.dot_colors = {
            'player': (255, 255, 255),  # White dot (player)
            'npc': (255, 255, 0),       # Yellow dots (NPCs)
            'player_other': (0, 255, 0)  # Green dots (other players)
        }
    
    def analyze_minimap(self, image: np.ndarray, 
                       region: Optional[Tuple[int, int, int, int]] = None) -> MinimapInfo:
        """
        Analyze minimap for navigation and situational awareness
        
        Args:
            image: Full game screenshot
            region: Minimap region (x, y, width, height)
            
        Returns:
            MinimapInfo with analysis results
        """
        try:
            # Extract minimap region
            minimap_region = region or self.minimap_region
            x, y, w, h = minimap_region
            minimap = image[y:y+h, x:x+w]
            
            info = MinimapInfo()
            
            # Find player position (center of minimap)
            info.player_position = (w // 2, h // 2)
            
            # Detect dots on minimap
            info.visible_npcs = self._detect_minimap_dots(minimap, 'npc')
            info.visible_players = self._detect_minimap_dots(minimap, 'player_other')
            
            # Analyze compass direction
            info.north_direction = self._detect_compass_direction(image)
            
            # Identify region type
            info.region_type = self._classify_region(minimap)
            
            # Find points of interest
            info.points_of_interest = self._find_poi(minimap)
            
            return info
            
        except Exception as e:
            logger.error(f"Minimap analysis failed: {e}")
            return MinimapInfo()
    
    def _detect_minimap_dots(self, minimap: np.ndarray, dot_type: str) -> List[Tuple[int, int]]:
        """Detect colored dots on minimap"""
        if dot_type not in self.dot_colors:
            return []
        
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
            
            # Create mask for specific dot color
            color_bgr = self.dot_colors[dot_type]
            color_hsv = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            
            # Define HSV range
            lower = np.array([max(0, color_hsv[0] - 10), 50, 50])
            upper = np.array([min(179, color_hsv[0] + 10), 255, 255])
            
            mask = cv2.inRange(hsv, lower, upper)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            dots = []
            for contour in contours:
                if cv2.contourArea(contour) > 2:  # Filter tiny noise
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        dots.append((cx, cy))
            
            return dots
            
        except Exception as e:
            logger.error(f"Dot detection failed for {dot_type}: {e}")
            return []
    
    def _detect_compass_direction(self, image: np.ndarray) -> float:
        """Detect compass direction (North = 0 degrees)"""
        # This would analyze the compass needle position
        # For now, return 0 (facing North)
        return 0.0
    
    def _classify_region(self, minimap: np.ndarray) -> str:
        """Classify the type of region based on minimap colors"""
        # Analyze dominant colors and patterns
        # Different regions have different color schemes
        # For now, return unknown
        return "unknown"
    
    def _find_poi(self, minimap: np.ndarray) -> List[Dict[str, Any]]:
        """Find points of interest on minimap (banks, shops, etc.)"""
        # This would identify special icons/markers
        return []


class SceneClassifier:
    """Classify the current game scene/activity"""
    
    def __init__(self):
        self.scene_indicators = {
            SceneType.COMBAT: [
                'health_bar_visible', 'combat_interface', 'damage_numbers',
                'combat_npcs', 'special_attack_bar'
            ],
            SceneType.SKILLING: [
                'skill_interface', 'resource_nodes', 'tools_equipped',
                'xp_drops', 'skilling_npcs'
            ],
            SceneType.BANKING: [
                'bank_interface', 'banker_npc', 'bank_pin_interface',
                'deposit_buttons', 'withdraw_buttons'
            ],
            SceneType.TRADING: [
                'trade_interface', 'grand_exchange', 'player_trading',
                'offer_interface', 'price_checker'
            ],
            SceneType.QUESTING: [
                'quest_interface', 'dialog_box', 'quest_npcs',
                'objective_text', 'quest_items'
            ]
        }
    
    def classify_scene(self, game_state_data: Dict[str, Any]) -> Tuple[SceneType, float]:
        """
        Classify the current scene based on available data
        
        Args:
            game_state_data: Dictionary with detection results
            
        Returns:
            Tuple of (scene_type, confidence)
        """
        try:
            scene_scores = {}
            
            # Analyze each potential scene type
            for scene_type, indicators in self.scene_indicators.items():
                score = self._calculate_scene_score(game_state_data, indicators)
                scene_scores[scene_type] = score
            
            # Find the best match
            if scene_scores:
                best_scene = max(scene_scores, key=scene_scores.get)
                confidence = scene_scores[best_scene]
                
                # Require minimum confidence
                if confidence > 0.3:
                    return best_scene, confidence
            
            return SceneType.UNKNOWN, 0.0
            
        except Exception as e:
            logger.error(f"Scene classification failed: {e}")
            return SceneType.UNKNOWN, 0.0
    
    def _calculate_scene_score(self, data: Dict[str, Any], indicators: List[str]) -> float:
        """Calculate confidence score for a scene type"""
        score = 0.0
        total_weight = len(indicators)
        
        if total_weight == 0:
            return 0.0
        
        # Check each indicator
        for indicator in indicators:
            if self._check_indicator(data, indicator):
                score += 1.0
        
        return score / total_weight
    
    def _check_indicator(self, data: Dict[str, Any], indicator: str) -> bool:
        """Check if a specific scene indicator is present"""
        # This would check various aspects of the game state
        # For now, return False for all indicators
        return False


class IntelligentVision:
    """
    Next-generation computer vision with AI capabilities
    
    Phase 2 implementation combining:
    - YOLO object detection
    - OCR text recognition  
    - Minimap analysis
    - Scene classification
    - Comprehensive game state analysis
    """
    
    def __init__(self, yolo_model_path: Optional[str] = None, device: str = "auto"):
        """
        Initialize the intelligent vision system
        
        Args:
            yolo_model_path: Path to custom YOLO model
            device: Device for AI inference ('cpu', 'cuda', 'auto')
        """
        self.device = device
        
        # Initialize detection components
        self.yolo_detector = YOLODetector(model_path=yolo_model_path, device=device)
        self.ocr_detector = OCRDetector(use_gpu=(device != 'cpu'))
        self.minimap_analyzer = MinimapAnalyzer()
        self.scene_classifier = SceneClassifier()
        
        # Game regions (these would be calibrated per client)
        self.game_regions = {
            'minimap': (570, 9, 146, 151),
            'chat': (7, 345, 506, 143),
            'inventory': (548, 205, 186, 262),
            'main_screen': (4, 4, 512, 334),
            'interface': (0, 0, 765, 503)  # Full interface
        }
        
        # Performance tracking
        self.stats = {
            'frames_processed': 0,
            'avg_processing_time': 0.0,
            'last_analysis_time': 0.0
        }
        
        logger.info("IntelligentVision system initialized (Phase 2)")
    
    def analyze_game_state(self, screenshot: np.ndarray) -> GameState:
        """
        Comprehensive game state analysis
        
        This is the main method that combines all detection capabilities
        to provide a complete understanding of the current game state.
        
        Args:
            screenshot: Full game screenshot (BGR format)
            
        Returns:
            GameState object with comprehensive analysis
        """
        start_time = time.time()
        
        try:
            # Initialize game state
            game_state = GameState(
                timestamp=start_time,
                scene_type=SceneType.UNKNOWN,
                confidence=0.0,
                player_status=PlayerStatus(),
                minimap=MinimapInfo(),
                inventory=InventoryInfo(),
                interface_state=InterfaceState()
            )
            
            # Parallel detection processes
            detection_results = {}
            
            # 1. YOLO Object Detection
            yolo_detections = self.yolo_detector.detect_objects(screenshot)
            detection_results['yolo'] = yolo_detections
            
            # Categorize YOLO detections
            for detection in yolo_detections:
                if detection.object_type == 'npc':
                    game_state.npcs.append(detection)
                elif detection.object_type == 'item':
                    game_state.items.append(detection)
                elif detection.object_type == 'player':
                    game_state.players.append(detection)
                elif detection.object_type == 'ui':
                    game_state.ui_elements.append(detection)
                elif detection.object_type == 'environment':
                    game_state.environment.append(detection)
            
            # 2. OCR Text Recognition
            text_detections = self.ocr_detector.detect_text(screenshot)
            detection_results['ocr'] = text_detections
            
            # Process text for interface state
            chat_messages = self.ocr_detector.read_chat_messages(
                screenshot, self.game_regions['chat']
            )
            game_state.interface_state.active_chat = chat_messages
            
            interface_text = self.ocr_detector.read_interface_text(
                screenshot, self.game_regions['interface']
            )
            game_state.interface_state.clickable_elements = interface_text.get('clickable_elements', [])
            
            # 3. Minimap Analysis
            game_state.minimap = self.minimap_analyzer.analyze_minimap(
                screenshot, self.game_regions['minimap']
            )
            
            # 4. Scene Classification
            scene_data = self._prepare_scene_data(detection_results, game_state)
            scene_type, scene_confidence = self.scene_classifier.classify_scene(scene_data)
            game_state.scene_type = scene_type
            game_state.confidence = scene_confidence
            
            # 5. Player Status Analysis
            game_state.player_status = self._analyze_player_status(screenshot, detection_results)
            
            # 6. Inventory Analysis
            game_state.inventory = self._analyze_inventory(screenshot, detection_results)
            
            # Calculate processing time and update stats
            processing_time = time.time() - start_time
            game_state.processing_time = processing_time
            
            self._update_performance_stats(processing_time)
            
            logger.debug(f"Game state analysis completed in {processing_time:.3f}s")
            logger.debug(f"Scene: {scene_type.value}, Objects: {len(yolo_detections)}, Text: {len(text_detections)}")
            
            return game_state
            
        except Exception as e:
            logger.error(f"Game state analysis failed: {e}")
            # Return minimal game state on error
            return GameState(
                timestamp=start_time,
                scene_type=SceneType.UNKNOWN,
                confidence=0.0,
                player_status=PlayerStatus(),
                minimap=MinimapInfo(),
                inventory=InventoryInfo(),
                interface_state=InterfaceState(),
                processing_time=time.time() - start_time
            )
    
    def _prepare_scene_data(self, detection_results: Dict[str, Any], 
                           game_state: GameState) -> Dict[str, Any]:
        """Prepare data for scene classification"""
        return {
            'yolo_detections': detection_results.get('yolo', []),
            'text_detections': detection_results.get('ocr', []),
            'npcs_count': len(game_state.npcs),
            'items_count': len(game_state.items),
            'players_count': len(game_state.players),
            'ui_elements_count': len(game_state.ui_elements),
            'chat_messages': game_state.interface_state.active_chat,
            'clickable_elements': game_state.interface_state.clickable_elements
        }
    
    def _analyze_player_status(self, screenshot: np.ndarray, 
                              detection_results: Dict[str, Any]) -> PlayerStatus:
        """Analyze player's current status"""
        status = PlayerStatus()
        
        # Analyze health/prayer/energy bars using color detection
        # This would examine the status orbs in the interface
        
        # Check for combat indicators
        combat_indicators = ['combat', 'enemy', 'damage', 'health_bar']
        for detection in detection_results.get('yolo', []):
            if any(indicator in detection.label.lower() for indicator in combat_indicators):
                status.is_in_combat = True
                break
        
        return status
    
    def _analyze_inventory(self, screenshot: np.ndarray,
                          detection_results: Dict[str, Any]) -> InventoryInfo:
        """Analyze inventory contents"""
        inventory = InventoryInfo()
        
        # Read item names from inventory region
        items = self.ocr_detector.read_item_names(
            screenshot, self.game_regions['inventory']
        )
        
        inventory.items = items
        inventory.free_slots = max(0, 28 - len(items))
        
        # Categorize items
        for item in items:
            if item.get('is_valuable', False):
                inventory.valuable_items.append(item['name'])
            
            # Check for consumables (food, potions, etc.)
            item_name = item['name'].lower()
            if any(keyword in item_name for keyword in ['food', 'potion', 'drink', 'eat']):
                inventory.consumables.append(item['name'])
        
        return inventory
    
    def _update_performance_stats(self, processing_time: float) -> None:
        """Update performance statistics"""
        self.stats['frames_processed'] += 1
        self.stats['last_analysis_time'] = processing_time
        
        # Update running average
        frames = self.stats['frames_processed']
        current_avg = self.stats['avg_processing_time']
        self.stats['avg_processing_time'] = (current_avg * (frames - 1) + processing_time) / frames
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return self.stats.copy()
    
    def calibrate_regions(self, screenshot: np.ndarray) -> bool:
        """
        Calibrate game regions for the current client setup
        This would use template matching to find interface elements
        """
        try:
            # This would implement automatic region detection
            # For now, use default regions
            logger.info("Using default game regions (calibration not implemented)")
            return True
        except Exception as e:
            logger.error(f"Region calibration failed: {e}")
            return False
    
    def visualize_analysis(self, screenshot: np.ndarray, 
                          game_state: GameState,
                          show_all: bool = True) -> np.ndarray:
        """
        Create visualization of the game state analysis
        
        Args:
            screenshot: Original screenshot
            game_state: Analysis results
            show_all: Whether to show all detection types
            
        Returns:
            Annotated image with analysis visualization
        """
        vis_image = screenshot.copy()
        
        try:
            # Draw YOLO detections
            if show_all:
                all_detections = (game_state.npcs + game_state.items + 
                                game_state.players + game_state.ui_elements + 
                                game_state.environment)
                vis_image = self.yolo_detector.visualize_detections(
                    vis_image, all_detections, show_priority=True
                )
            
            # Draw minimap info
            if game_state.minimap.player_position:
                minimap_region = self.game_regions['minimap']
                x, y, w, h = minimap_region
                
                # Draw minimap border
                cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 255), 2)
                
                # Draw NPC dots
                for npc_pos in game_state.minimap.visible_npcs:
                    dot_x, dot_y = npc_pos
                    cv2.circle(vis_image, (x + dot_x, y + dot_y), 3, (0, 255, 0), -1)
            
            # Add scene classification text
            scene_text = f"Scene: {game_state.scene_type.value} ({game_state.confidence:.2f})"
            cv2.putText(vis_image, scene_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add performance info
            perf_text = f"Processing: {game_state.processing_time:.3f}s"
            cv2.putText(vis_image, perf_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add object counts
            counts_text = f"NPCs: {len(game_state.npcs)}, Items: {len(game_state.items)}, Players: {len(game_state.players)}"
            cv2.putText(vis_image, counts_text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            return vis_image
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
            return vis_image


# Global intelligent vision instance
intelligent_vision = IntelligentVision()