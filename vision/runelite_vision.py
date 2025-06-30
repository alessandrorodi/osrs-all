#!/usr/bin/env python3
"""
RuneLite-Specific Intelligent Vision System
Properly detects RuneLite UI elements, orbs, tabs, and content
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class RuneLiteTab(Enum):
    COMBAT = "combat"
    STATS = "stats" 
    QUEST = "quest"
    INVENTORY = "inventory"
    EQUIPMENT = "equipment"
    PRAYER = "prayer"
    MAGIC = "magic"
    CLAN = "clan"
    ACCOUNT = "account"
    LOGOUT = "logout"
    OPTIONS = "options"
    EMOTES = "emotes"
    MUSIC = "music"
    UNKNOWN = "unknown"

@dataclass
class RuneLiteElement:
    name: str
    position: Tuple[int, int]
    confidence: float
    element_type: str
    data: Any = None

class RuneLiteVision:
    """
    RuneLite-Specific Vision System
    Accurately detects RuneLite interface elements based on actual layout
    """
    
    def __init__(self):
        # RuneLite-specific UI patterns
        self.runelite_patterns = self._init_runelite_patterns()
        
        # Performance tracking
        self.analysis_times = []
        
        # RuneLite knowledge
        self.runelite_knowledge = self._init_runelite_knowledge()
        
    def _init_runelite_patterns(self) -> Dict[str, Dict]:
        """Initialize RuneLite-specific visual patterns focusing on IN-GAME OSRS UI"""
        return {
            # Define the game viewport area (excludes RuneLite client UI)
            'game_viewport': {
                'region_ratio': (0.0, 0.0, 0.85, 1.0),  # Left 85% is game area
            },
            
            # IN-GAME OSRS Orbs (within game viewport, top-right area)
            'health_orb': {
                'region_ratio': (0.73, 0.05, 0.08, 0.08),  # Top-right of game area
                'color_ranges': {
                    'high_health': [(0, 100, 0), (100, 255, 100)],  # Green
                    'medium_health': [(0, 100, 100), (100, 255, 255)], # Yellow  
                    'low_health': [(0, 0, 100), (100, 100, 255)]      # Red
                }
            },
            'prayer_orb': {
                'region_ratio': (0.73, 0.15, 0.08, 0.08),  # Below health orb
                'color_ranges': {
                    'high_prayer': [(100, 0, 0), (255, 100, 100)],    # Blue
                    'low_prayer': [(50, 50, 50), (150, 150, 150)]     # Gray
                }
            },
            'energy_orb': {
                'region_ratio': (0.73, 0.25, 0.08, 0.08),  # Below prayer orb
                'color_ranges': {
                    'high_energy': [(0, 200, 200), (100, 255, 255)],  # Yellow
                    'low_energy': [(50, 50, 50), (150, 150, 150)]     # Gray
                }
            },
            'special_orb': {
                'region_ratio': (0.73, 0.35, 0.08, 0.08),  # Below energy orb
                'color_ranges': {
                    'special_ready': [(0, 150, 150), (100, 255, 255)], # Bright
                    'special_charging': [(50, 50, 50), (150, 150, 150)] # Dark
                }
            },
            
            # IN-GAME OSRS Tab detection (bottom of right interface panel)
            'ingame_tab_strip': {
                'region_ratio': (0.55, 0.85, 0.25, 0.10),  # Bottom strip of interface panel
                'tab_positions': {
                    # OSRS tabs arranged horizontally at bottom of interface
                    'combat': (0.565, 0.90),     # Attack options (leftmost)
                    'stats': (0.595, 0.90),      # Stats  
                    'quest': (0.625, 0.90),      # Quest
                    'inventory': (0.655, 0.90),  # Inventory (backpack)
                    'equipment': (0.685, 0.90),  # Equipment (armor)
                    'prayer': (0.715, 0.90),     # Prayer (hands)
                    'magic': (0.745, 0.90),      # Magic (star)
                    'friends': (0.775, 0.90),    # Friends (rightmost)
                }
            },
            
            # IN-GAME Content areas (within game viewport)
            'ingame_inventory': {
                'region_ratio': (0.55, 0.35, 0.25, 0.30),  # Right side of game viewport
                'grid_size': (4, 7),  # 4 columns, 7 rows
            },
            'ingame_prayer_book': {
                'region_ratio': (0.55, 0.25, 0.25, 0.40),  # When prayer tab selected
                'grid_size': (5, 6),  # 5 columns, 6 rows of prayers
            },
            'ingame_equipment': {
                'region_ratio': (0.55, 0.20, 0.25, 0.45),  # When equipment tab selected
            },
            
            # IN-GAME Chat area (bottom of game viewport)
            'ingame_chat': {
                'region_ratio': (0.01, 0.75, 0.78, 0.24),  # Bottom of game area
                'text_color_range': [(200, 200, 200), (255, 255, 255)]
            }
        }
    
    def _init_runelite_knowledge(self) -> Dict[str, Any]:
        """Initialize RuneLite game knowledge"""
        return {
            'tab_icons': {
                'inventory': 'backpack icon',
                'equipment': 'armor icon', 
                'prayer': 'praying hands icon',
                'magic': 'star icon',
                'combat': 'crossed swords icon'
            },
            'orb_meanings': {
                'health': 'Player hitpoints',
                'prayer': 'Prayer points',
                'energy': 'Run energy', 
                'special': 'Special attack energy'
            }
        }
    
    def analyze_runelite_screenshot(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Main RuneLite analysis function
        Detects all important RuneLite UI elements accurately
        """
        start_time = time.time()
        
        try:
            h, w = screenshot.shape[:2]
            logger.debug(f"Analyzing RuneLite screenshot: {w}x{h}")
            
            results = {
                'active_tab': RuneLiteTab.UNKNOWN,
                'orbs': {},
                'inventory_items': [],
                'prayers': [],
                'equipment': {},
                'chat_messages': [],
                'ui_elements': []
            }
            
            # Step 1: Detect active tab (most important for context)
            active_tab = self._detect_active_tab(screenshot)
            results['active_tab'] = active_tab
            
            # Step 2: Always detect orbs (critical game info)
            results['orbs'] = self._detect_orbs(screenshot)
            
            # Step 3: Context-aware content detection
            if active_tab == RuneLiteTab.INVENTORY:
                results['inventory_items'] = self._detect_inventory_items(screenshot)
            elif active_tab == RuneLiteTab.PRAYER:
                results['prayers'] = self._detect_prayers(screenshot)
            elif active_tab == RuneLiteTab.EQUIPMENT:
                results['equipment'] = self._detect_equipment(screenshot)
            
            # Step 4: Always check chat
            results['chat_messages'] = self._detect_chat_activity(screenshot)
            
            # Add performance metrics
            processing_time = time.time() - start_time
            self.analysis_times.append(processing_time)
            if len(self.analysis_times) > 100:
                self.analysis_times.pop(0)
            
            results['performance'] = {
                'processing_time': processing_time,
                'avg_processing_time': sum(self.analysis_times) / len(self.analysis_times),
                'active_tab': active_tab.value,
                'screenshot_size': f"{w}x{h}"
            }
            
            logger.debug(f"RuneLite analysis completed in {processing_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"RuneLite vision analysis failed: {e}")
            return self._empty_results()
    
    def _detect_active_tab(self, screenshot: np.ndarray) -> RuneLiteTab:
        """
        Detect which IN-GAME OSRS tab is currently selected
        Focuses on the game viewport, not RuneLite client tabs
        """
        h, w = screenshot.shape[:2]
        tab_strip = self.runelite_patterns['ingame_tab_strip']
        
        # Get the in-game tab area (bottom-right of game viewport)
        tab_region = self._get_region_by_ratio(screenshot, tab_strip['region_ratio'])
        
        if tab_region.size == 0:
            return RuneLiteTab.UNKNOWN
        
        highest_brightness = 0
        selected_tab = RuneLiteTab.UNKNOWN
        
        # Check each in-game tab position
        for tab_name, (x_ratio, y_ratio) in tab_strip['tab_positions'].items():
            # Calculate tab position within the screenshot
            tab_x = int(w * x_ratio)
            tab_y = int(h * y_ratio)
            
            # Define tab region (small area around the tab)
            tab_size = 30  # Tab icon size in pixels
            start_x = max(0, tab_x - tab_size//2)
            end_x = min(w, tab_x + tab_size//2)
            start_y = max(0, tab_y - tab_size//2)
            end_y = min(h, tab_y + tab_size//2)
            
            tab_icon_region = screenshot[start_y:end_y, start_x:end_x]
            
            if tab_icon_region.size > 0:
                # Selected in-game tabs are typically brighter/highlighted
                brightness = np.mean(tab_icon_region)
                
                if brightness > highest_brightness and brightness > 80:  # Higher threshold for in-game tabs
                    highest_brightness = brightness
                    try:
                        selected_tab = RuneLiteTab(tab_name)
                    except ValueError:
                        if tab_name == 'friends':
                            selected_tab = RuneLiteTab.CLAN  # Map friends to clan
                        elif tab_name == 'ignore':
                            selected_tab = RuneLiteTab.OPTIONS  # Map ignore to options
                        else:
                            selected_tab = RuneLiteTab.UNKNOWN
        
        return selected_tab
    
    def _detect_orbs(self, screenshot: np.ndarray) -> Dict[str, Dict]:
        """
        Detect all orbs and their states
        Returns health, prayer, energy, special attack info
        """
        orbs = {}
        
        orb_types = ['health_orb', 'prayer_orb', 'energy_orb', 'special_orb']
        
        for orb_type in orb_types:
            orb_pattern = self.runelite_patterns[orb_type]
            orb_region = self._get_region_by_ratio(screenshot, orb_pattern['region_ratio'])
            
            if orb_region.size > 0:
                orb_name = orb_type.replace('_orb', '')
                orbs[orb_name] = self._analyze_orb_state(orb_region, orb_pattern)
        
        return orbs
    
    def _analyze_orb_state(self, orb_region: np.ndarray, orb_pattern: Dict) -> Dict[str, Any]:
        """
        Analyze individual orb state using color analysis
        """
        try:
            # Convert to HSV for better color detection
            hsv = cv2.cvtColor(orb_region, cv2.COLOR_BGR2HSV)
            
            # Check each color range to determine orb state
            for state_name, color_range in orb_pattern['color_ranges'].items():
                lower = np.array(color_range[0])
                upper = np.array(color_range[1])
                
                mask = cv2.inRange(hsv, lower, upper)
                percentage = np.sum(mask > 0) / mask.size
                
                if percentage > 0.1:  # 10% of orb shows this color
                    return {
                        'state': state_name,
                        'percentage': int(percentage * 100),
                        'confidence': 0.8
                    }
            
            # Default state if no color match
            return {
                'state': 'unknown',
                'percentage': 0,
                'confidence': 0.5
            }
            
        except Exception as e:
            logger.debug(f"Orb analysis failed: {e}")
            return {'state': 'error', 'percentage': 0, 'confidence': 0.0}
    
    def _detect_inventory_items(self, screenshot: np.ndarray) -> List[Dict]:
        """
        Detect items in IN-GAME inventory when inventory tab is active
        """
        items = []
        
        inv_pattern = self.runelite_patterns['ingame_inventory']
        inv_region = self._get_region_by_ratio(screenshot, inv_pattern['region_ratio'])
        
        if inv_region.size == 0:
            return items
        
        # Analyze 4x7 in-game inventory grid
        grid_w, grid_h = inv_pattern['grid_size']
        slot_w = inv_region.shape[1] // grid_w
        slot_h = inv_region.shape[0] // grid_h
        
        for row in range(grid_h):
            for col in range(grid_w):
                x = col * slot_w
                y = row * slot_h
                slot = inv_region[y:y+slot_h, x:x+slot_w]
                
                item_detected = self._detect_item_in_slot(slot, row * grid_w + col)
                if item_detected:
                    items.append(item_detected)
        
        return items
    
    def _detect_prayers(self, screenshot: np.ndarray) -> List[Dict]:
        """
        Detect IN-GAME prayer states when prayer tab is active
        """
        prayers = []
        
        prayer_pattern = self.runelite_patterns['ingame_prayer_book']
        prayer_region = self._get_region_by_ratio(screenshot, prayer_pattern['region_ratio'])
        
        if prayer_region.size == 0:
            return prayers
        
        # Analyze in-game prayer grid (5x6 typically)
        grid_w, grid_h = prayer_pattern['grid_size']
        slot_w = prayer_region.shape[1] // grid_w
        slot_h = prayer_region.shape[0] // grid_h
        
        for row in range(grid_h):
            for col in range(grid_w):
                x = col * slot_w
                y = row * slot_h
                prayer_slot = prayer_region[y:y+slot_h, x:x+slot_w]
                
                prayer_state = self._detect_prayer_state(prayer_slot, row * grid_w + col)
                if prayer_state:
                    prayers.append(prayer_state)
        
        return prayers
    
    def _detect_equipment(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        Detect IN-GAME equipment when equipment tab is active
        """
        equipment = {}
        
        # Detect in-game equipment viewer content
        equip_pattern = self.runelite_patterns['ingame_equipment']
        equip_region = self._get_region_by_ratio(screenshot, equip_pattern['region_ratio'])
        
        if equip_region.size > 0:
            # Simple detection - check if there's varied content (not empty)
            color_variance = np.std(equip_region)
            
            if color_variance > 20:  # Has equipment items
                equipment['has_equipment'] = True
                equipment['confidence'] = min(1.0, color_variance / 50.0)
            else:
                equipment['has_equipment'] = False
                equipment['confidence'] = 0.8
        
        return equipment
    
    def _detect_chat_activity(self, screenshot: np.ndarray) -> List[Dict]:
        """
        Detect IN-GAME chat messages and activity
        """
        chat_messages = []
        
        chat_pattern = self.runelite_patterns['ingame_chat']
        chat_region = self._get_region_by_ratio(screenshot, chat_pattern['region_ratio'])
        
        if chat_region.size == 0:
            return chat_messages
        
        # Simple text detection using contour analysis
        gray = cv2.cvtColor(chat_region, cv2.COLOR_BGR2GRAY)
        
        # Find white text areas
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter for text-like contours
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Text-like dimensions
            if 20 < w < 300 and 10 < h < 30:
                chat_messages.append({
                    'type': 'in_game_chat',
                    'position': (x + w//2, y + h//2),
                    'size': (w, h),
                    'confidence': 0.7
                })
        
        return chat_messages
    
    # Helper methods
    def _get_region_by_ratio(self, screenshot: np.ndarray, ratio: Tuple[float, float, float, float]) -> np.ndarray:
        """Extract region using ratio coordinates"""
        h, w = screenshot.shape[:2]
        x, y, w_ratio, h_ratio = ratio
        
        start_x = max(0, int(w * x))
        start_y = max(0, int(h * y))
        end_x = min(w, int(w * (x + w_ratio)))
        end_y = min(h, int(h * (y + h_ratio)))
        
        return screenshot[start_y:end_y, start_x:end_x]
    
    def _detect_item_in_slot(self, slot: np.ndarray, slot_number: int) -> Optional[Dict]:
        """Detect if inventory slot contains an item"""
        if slot.size == 0:
            return None
        
        # Use color variance to detect items
        color_std = np.std(slot)
        mean_color = np.mean(slot.reshape(-1, 3), axis=0)
        
        # Items have more color variance than empty slots
        if color_std >= 15 and np.mean(mean_color) >= 50:
            return {
                'slot': slot_number,
                'name': f'Item_Slot_{slot_number}',
                'confidence': min(0.9, color_std / 50.0),
                'type': 'inventory_item',
                'color_variance': color_std
            }
        
        return None
    
    def _detect_prayer_state(self, prayer_slot: np.ndarray, prayer_number: int) -> Optional[Dict]:
        """Detect if prayer is active/inactive"""
        if prayer_slot.size == 0:
            return None
        
        # Active prayers are typically brighter
        brightness = np.mean(prayer_slot)
        
        if brightness > 80:  # Active prayer
            return {
                'prayer_id': prayer_number,
                'name': f'Prayer_{prayer_number}',
                'active': True,
                'confidence': min(1.0, brightness / 120.0),
                'type': 'prayer'
            }
        elif brightness > 40:  # Available but inactive
            return {
                'prayer_id': prayer_number,
                'name': f'Prayer_{prayer_number}',
                'active': False,
                'confidence': 0.8,
                'type': 'prayer'
            }
        
        return None
    
    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure"""
        return {
            'active_tab': RuneLiteTab.UNKNOWN,
            'orbs': {},
            'inventory_items': [],
            'prayers': [],
            'equipment': {},
            'chat_messages': [],
            'ui_elements': []
        }

# Global RuneLite vision instance
runelite_vision = RuneLiteVision() 