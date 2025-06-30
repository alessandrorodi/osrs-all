#!/usr/bin/env python3
"""
Intelligent AI Vision System for OSRS
Combines game bot efficiency with AI intelligence for instant analysis
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class UIState(Enum):
    INVENTORY_TAB = "inventory"
    STATS_TAB = "stats"
    COMBAT_TAB = "combat"
    PRAYER_TAB = "prayer"
    MAGIC_TAB = "magic"
    CHAT_FOCUSED = "chat"
    UNKNOWN = "unknown"

@dataclass
class VisualElement:
    name: str
    position: Tuple[int, int]
    confidence: float
    element_type: str
    data: Any = None

class IntelligentVision:
    """
    AI-Powered Real-Time OSRS Vision System
    
    Philosophy:
    - Think like a human: Focus attention on important areas
    - Act like a game bot: Use efficient pattern recognition
    - Process like AI: Understand context and make smart decisions
    """
    
    def __init__(self):
        # Visual patterns for instant recognition
        self.ui_patterns = self._init_ui_patterns()
        
        # State tracking for efficiency
        self.current_ui_state = UIState.UNKNOWN
        self.last_screenshot = None
        self.last_analysis_time = 0
        
        # Performance tracking
        self.analysis_times = []
        
        # OSRS knowledge base
        self.osrs_knowledge = self._init_osrs_knowledge()
        
    def _init_ui_patterns(self) -> Dict[str, Dict]:
        """Initialize visual patterns for instant UI recognition"""
        return {
            # Tab detection patterns (color ranges for selected tabs)
            'inventory_tab_selected': {
                'color_range': [(139, 105, 74), (160, 125, 94)],  # Brown selected tab
                'position_ratio': (0.85, 0.25),  # Approx position in UI
                'size': (25, 20)
            },
            'stats_tab_selected': {
                'color_range': [(139, 105, 74), (160, 125, 94)],
                'position_ratio': (0.85, 0.20),
                'size': (25, 20)
            },
            # Chat patterns
            'chat_text': {
                'color_range': [(200, 200, 200), (255, 255, 255)],  # White text
                'background': [(0, 0, 0), (50, 50, 50)],  # Dark background
                'region_ratio': (0.0, 0.75, 0.65, 0.25)  # Bottom portion
            },
            # Inventory grid
            'inventory_slots': {
                'grid_size': (4, 7),  # 4x7 grid
                'region_ratio': (0.73, 0.30, 0.26, 0.40),
                'slot_size': (36, 32)
            },
            # Health/Prayer orbs
            'health_orb': {
                'color_range': [(0, 150, 0), (50, 255, 50)],  # Green health
                'position_ratio': (0.1, 0.1),
                'size': (25, 25)
            }
        }
    
    def _init_osrs_knowledge(self) -> Dict[str, Any]:
        """Initialize OSRS game knowledge for intelligent analysis"""
        return {
            'common_items': {
                # Items have distinctive colors/shapes
                'coins': {'color': (255, 215, 0), 'keywords': ['gp', 'coins']},
                'food': {'colors': [(255, 0, 0), (255, 165, 0)], 'keywords': ['lobster', 'shark', 'bread']},
                'weapons': {'colors': [(128, 128, 128), (192, 192, 192)], 'keywords': ['sword', 'bow', 'staff']},
            },
            'ui_context': {
                'inventory_active': 'Look for items in 4x7 grid',
                'stats_active': 'Look for skill levels and XP',
                'chat_active': 'Parse recent messages',
            }
        }
    
    def analyze_screenshot_intelligent(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Main intelligent analysis function
        Process like human vision: fast, focused, context-aware
        """
        start_time = time.time()
        
        try:
            h, w = screenshot.shape[:2]
            
            # Step 1: INSTANT UI state detection (< 5ms)
            ui_state = self._detect_ui_state_fast(screenshot)
            
            # Step 2: FOCUSED attention based on UI state (< 20ms)
            focused_analysis = self._analyze_focused_regions(screenshot, ui_state)
            
            # Step 3: SMART interpretation using OSRS knowledge (< 10ms)
            intelligent_results = self._interpret_with_ai(focused_analysis, ui_state)
            
            # Performance tracking
            processing_time = time.time() - start_time
            self.analysis_times.append(processing_time)
            if len(self.analysis_times) > 100:
                self.analysis_times.pop(0)
            
            intelligent_results['performance'] = {
                'processing_time': processing_time,
                'avg_processing_time': sum(self.analysis_times) / len(self.analysis_times),
                'ui_state': ui_state.value,
                'screenshot_size': f"{w}x{h}"
            }
            
            logger.debug(f"Intelligent analysis completed in {processing_time:.3f}s")
            return intelligent_results
            
        except Exception as e:
            logger.error(f"Intelligent vision analysis failed: {e}")
            return self._empty_results()
    
    def _detect_ui_state_fast(self, screenshot: np.ndarray) -> UIState:
        """
        Instantly detect UI state using visual patterns
        Like human vision: immediate recognition of interface state
        """
        h, w = screenshot.shape[:2]
        
        # Check inventory tab (most common) - bottom right tab area
        inv_region = self._get_region_by_ratio(screenshot, (0.83, 0.24, 0.06, 0.05))
        if self._is_tab_selected(inv_region, 'inventory'):
            return UIState.INVENTORY_TAB
        
        # Check stats tab
        stats_region = self._get_region_by_ratio(screenshot, (0.83, 0.19, 0.06, 0.05))
        if self._is_tab_selected(stats_region, 'stats'):
            return UIState.STATS_TAB
        
        # Check if chat is active (typing or recent activity)
        chat_region = self._get_region_by_ratio(screenshot, (0.0, 0.85, 0.65, 0.15))
        if self._has_chat_activity(chat_region):
            return UIState.CHAT_FOCUSED
        
        return UIState.UNKNOWN
    
    def _analyze_focused_regions(self, screenshot: np.ndarray, ui_state: UIState) -> Dict[str, List[VisualElement]]:
        """
        Focused analysis based on UI state
        Like saccadic eye movements: only look where it matters
        """
        elements = {
            'items': [],
            'text': [],
            'ui_elements': [],
            'numbers': []
        }
        
        # Always check inventory for items (even if tab detection is uncertain)
        elements['items'] = self._analyze_inventory_fast(screenshot)
        
        # Always check for chat activity
        elements['text'] = self._analyze_chat_fast(screenshot)
        
        if ui_state == UIState.STATS_TAB:
            # Focus on skill levels when stats tab is selected
            elements['numbers'] = self._analyze_stats_fast(screenshot)
        
        # Always check for important UI elements (health, prayer)
        elements['ui_elements'] = self._analyze_critical_ui(screenshot)
        
        return elements
    
    def _analyze_inventory_fast(self, screenshot: np.ndarray) -> List[VisualElement]:
        """
        Fast inventory analysis using grid detection and pattern recognition
        Only processes when inventory tab is actually selected
        """
        items = []
        h, w = screenshot.shape[:2]
        
        # Get inventory region (lower right where inventory actually is)
        inv_region = self._get_region_by_ratio(screenshot, (0.73, 0.35, 0.26, 0.35))
        
        if inv_region.size == 0:
            return items
        
        # Detect items in 4x7 grid using color/pattern analysis
        grid_w, grid_h = 4, 7
        slot_w = inv_region.shape[1] // grid_w
        slot_h = inv_region.shape[0] // grid_h
        
        for row in range(grid_h):
            for col in range(grid_w):
                x = col * slot_w
                y = row * slot_h
                slot = inv_region[y:y+slot_h, x:x+slot_w]
                
                # Fast item detection using color analysis
                item_detected = self._detect_item_in_slot(slot)
                if item_detected:
                    global_x = int(w * 0.73) + x + slot_w // 2
                    global_y = int(h * 0.35) + y + slot_h // 2
                    
                    items.append(VisualElement(
                        name=item_detected['name'],
                        position=(global_x, global_y),
                        confidence=item_detected['confidence'],
                        element_type='item',
                        data={'slot': row * grid_w + col}
                    ))
        
        return items
    
    def _analyze_chat_fast(self, screenshot: np.ndarray) -> List[VisualElement]:
        """
        Fast chat analysis using minimal OCR
        """
        chat_elements = []
        
        # Get chat region (bottom area)
        chat_region = self._get_region_by_ratio(screenshot, (0.01, 0.75, 0.64, 0.24))
        
        if chat_region.size == 0:
            return chat_elements
        
        # Simple text detection without heavy OCR
        # Look for white text patterns on dark background
        gray = cv2.cvtColor(chat_region, cv2.COLOR_BGR2GRAY)
        
        # Find white text areas
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours that look like text
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Text-like dimensions
            if 10 < w < 200 and 8 < h < 25:
                h_screen, w_screen = screenshot.shape[:2]
                global_x = int(w_screen * 0.01) + x + w // 2
                global_y = int(h_screen * 0.75) + y + h // 2
                
                chat_elements.append(VisualElement(
                    name='text_area',
                    position=(global_x, global_y),
                    confidence=0.7,
                    element_type='text',
                    data={'bbox': (x, y, w, h)}
                ))
        
        return chat_elements
    
    def _analyze_stats_fast(self, screenshot: np.ndarray) -> List[VisualElement]:
        """
        Fast stats analysis using number pattern recognition
        """
        stats = []
        
        # Get stats region (right side when stats tab is selected)
        stats_region = self._get_region_by_ratio(screenshot, (0.73, 0.20, 0.26, 0.50))
        
        if stats_region.size == 0:
            return stats
        
        # Look for number-like patterns
        gray = cv2.cvtColor(stats_region, cv2.COLOR_BGR2GRAY)
        
        # Find bright areas (numbers are usually bright)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Number-like dimensions
            if 15 < w < 100 and 10 < h < 30:
                h_screen, w_screen = screenshot.shape[:2]
                global_x = int(w_screen * 0.73) + x + w // 2
                global_y = int(h_screen * 0.20) + y + h // 2
                
                stats.append(VisualElement(
                    name='stat_number',
                    position=(global_x, global_y),
                    confidence=0.8,
                    element_type='number',
                    data={'bbox': (x, y, w, h)}
                ))
        
        return stats
    
    def _analyze_critical_ui(self, screenshot: np.ndarray) -> List[VisualElement]:
        """
        Always check critical UI elements using fast color detection
        """
        ui_elements = []
        
        # Health orb check (top left area)
        health_region = self._get_region_by_ratio(screenshot, (0.08, 0.08, 0.06, 0.06))
        health_level = self._detect_health_level(health_region)
        
        if health_level > 0:
            h, w = screenshot.shape[:2]
            ui_elements.append(VisualElement(
                name='health',
                position=(int(w * 0.11), int(h * 0.11)),
                confidence=0.9,
                element_type='ui',
                data={'health_percent': health_level}
            ))
        
        return ui_elements
    
    def _interpret_with_ai(self, elements: Dict[str, List[VisualElement]], ui_state: UIState) -> Dict[str, Any]:
        """
        AI interpretation of visual elements using OSRS knowledge
        """
        results = {
            'chat_messages': [],
            'items': [],
            'player_stats': {},
            'ui_state': ui_state.value,
            'alerts': []
        }
        
        # Process items (always check, regardless of UI state)
        for item in elements['items']:
            item_data = {
                'name': f"Item_Slot_{item.data.get('slot', 0) if item.data else 0}",
                'position': item.position,
                'confidence': item.confidence,
                'slot': item.data.get('slot', -1) if item.data else -1
            }
            results['items'].append(item_data)
        
        # Process text elements 
        for text_elem in elements['text']:
            results['chat_messages'].append({
                'text': f'Text_detected_at_{text_elem.position}',
                'position': text_elem.position,
                'confidence': text_elem.confidence
            })
        
        # Process stats/numbers
        for num_elem in elements['numbers']:
            stat_name = f'stat_{len(results["player_stats"])}'
            results['player_stats'][stat_name] = {
                'position': num_elem.position,
                'confidence': num_elem.confidence
            }
        
        # AI-driven contextual alerts
        if len(results['items']) > 0:
            results['alerts'].append(f"Found {len(results['items'])} items in inventory")
        
        if len(results['chat_messages']) > 0:
            results['alerts'].append(f"Detected {len(results['chat_messages'])} chat elements")
        
        if ui_state == UIState.UNKNOWN:
            results['alerts'].append("UI state unclear - running general analysis")
        
        return results
    
    # Helper methods
    def _get_region_by_ratio(self, screenshot: np.ndarray, ratio: Tuple[float, float, float, float]) -> np.ndarray:
        """Extract region using ratio coordinates"""
        h, w = screenshot.shape[:2]
        x, y, w_ratio, h_ratio = ratio
        
        start_x = int(w * x)
        start_y = int(h * y)
        end_x = int(w * (x + w_ratio))
        end_y = int(h * (y + h_ratio))
        
        # Ensure bounds are valid
        start_x = max(0, start_x)
        start_y = max(0, start_y)
        end_x = min(w, end_x)
        end_y = min(h, end_y)
        
        return screenshot[start_y:end_y, start_x:end_x]
    
    def _is_tab_selected(self, region: np.ndarray, tab_type: str) -> bool:
        """Fast tab selection detection using color analysis"""
        if region.size == 0:
            return False
        
        # Check for highlighted/selected tab appearance
        # Selected tabs are typically brighter or have different color
        
        mean_brightness = np.mean(region)
        
        # Selected tabs are usually brighter than unselected
        return mean_brightness > 50  # Lower threshold for RuneLite tabs
    
    def _has_chat_activity(self, region: np.ndarray) -> bool:
        """Detect if chat has recent activity"""
        if region.size == 0:
            return False
        
        # Look for white text on dark background
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        white_pixels = np.sum(gray > 200)
        total_pixels = gray.size
        
        return (white_pixels / total_pixels) > 0.02  # 2% white pixels suggests text (lowered for RuneLite)
    
    def _detect_item_in_slot(self, slot: np.ndarray) -> Optional[Dict[str, Any]]:
        """Fast item detection using visual analysis"""
        if slot.size == 0:
            return None
        
        # Calculate color variance - empty slots are uniform
        color_std = np.std(slot)
        
        if color_std < 15:  # Too uniform = likely empty
            return None
        
        # Check for non-background colors
        mean_color = np.mean(slot.reshape(-1, 3), axis=0)
        
        # Empty/background is typically dark gray/brown
        if np.all(mean_color < 70):  # Too dark = likely empty
            return None
        
        # If we get here, likely has an item
        return {
            'name': 'detected_item',
            'confidence': min(0.9, color_std / 50.0)  # Higher variance = higher confidence
        }
    
    def _detect_health_level(self, region: np.ndarray) -> int:
        """Detect health level using color analysis"""
        if region.size == 0:
            return 0
        
        # Look for green/red colors in health orb
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        
        # Green (healthy) color range
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Red (low health) color range  
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, lower_red, upper_red)
        
        green_percentage = np.sum(green_mask > 0) / green_mask.size
        red_percentage = np.sum(red_mask > 0) / red_mask.size
        
        if green_percentage > 0.1:
            return int(green_percentage * 100)
        elif red_percentage > 0.1:
            return int(red_percentage * 50)  # Low health
        
        return 0
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure"""
        return {
            'chat_messages': [],
            'items': [],
            'player_stats': {},
            'ui_state': 'unknown',
            'alerts': []
        }

# Global intelligent vision instance
intelligent_vision = IntelligentVision()