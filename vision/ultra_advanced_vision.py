#!/usr/bin/env python3
"""
🚀🧠 ULTRA-ADVANCED RuneLite Vision System 
Fixes viewport detection and uses content-aware tab detection
"""

import cv2
import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from enum import Enum
import logging

# Advanced ML libraries
try:
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

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
    UNKNOWN = "unknown"

class UltraAdvancedRuneLiteVision:
    """
    🚀 ULTRA-ADVANCED Computer Vision for RuneLite
    FIXES: Proper viewport detection + Content-aware tab detection
    """
    
    def __init__(self):
        self.analysis_times = []
        logger.info("🚀🧠 ULTRA-ADVANCED RuneLite Vision initialized")
        logger.info(f"🔬 ML Libraries: sklearn={HAS_SKLEARN}")
        
    def analyze_runelite_ultra(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        🚀 ULTRA-ADVANCED analysis with CORRECTED viewport detection
        """
        start_time = time.time()
        
        try:
            h, w = screenshot.shape[:2]
            logger.debug(f"🔬 ULTRA analysis: {w}x{h}")
            
            results = {
                'active_tab': RuneLiteTab.UNKNOWN,
                'orbs': {},
                'inventory_items': [],
                'chat_messages': [],
                'detection_method': 'ultra_advanced',
                'confidence_scores': {}
            }
            
            # 🧠 CONTENT-AWARE TAB DETECTION (instead of brightness)
            tab_result = self._detect_active_tab_by_content(screenshot)
            results['active_tab'] = tab_result['tab']
            results['confidence_scores']['tab_detection'] = tab_result['confidence']
            
            # 🔮 CORRECTED ORB DETECTION  
            orbs_result = self._detect_orbs_corrected(screenshot)
            results['orbs'] = orbs_result['orbs']
            results['confidence_scores']['orb_detection'] = orbs_result['confidence']
            
            # 📦 SMART INVENTORY DETECTION
            if tab_result['tab'] == RuneLiteTab.INVENTORY:
                inv_result = self._detect_inventory_smart(screenshot)
                results['inventory_items'] = inv_result['items']
                results['confidence_scores']['inventory'] = inv_result['confidence']
            
            # 💬 CORRECTED CHAT DETECTION
            chat_result = self._detect_chat_corrected(screenshot)
            results['chat_messages'] = chat_result['messages']
            results['confidence_scores']['chat'] = chat_result['confidence']
            
            # Performance metrics
            processing_time = time.time() - start_time
            self.analysis_times.append(processing_time)
            
            results['performance'] = {
                'processing_time': processing_time,
                'method': 'ultra_advanced_cv',
                'total_confidence': np.mean(list(results['confidence_scores'].values()))
            }
            
            logger.debug(f"🎉 ULTRA analysis: {processing_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"❌ ULTRA analysis failed: {e}")
            return self._empty_results()
    
    def _detect_active_tab_by_content(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """
        🧠 CONTENT-AWARE tab detection - Look at what's displayed, not tab brightness
        """
        h, w = screenshot.shape[:2]
        
        # Check for inventory grid pattern (4x7 item slots)
        inventory_confidence = self._check_inventory_content(screenshot)
        
        # Check for prayer book pattern (prayer icons)
        prayer_confidence = self._check_prayer_content(screenshot)
        
        # Check for equipment pattern (character model)
        equipment_confidence = self._check_equipment_content(screenshot)
        
        # Check for combat options (attack style buttons)
        combat_confidence = self._check_combat_content(screenshot)
        
        # Check for stats (skill icons and numbers)
        stats_confidence = self._check_stats_content(screenshot)
        
        # Find highest confidence tab
        confidence_scores = {
            RuneLiteTab.INVENTORY: inventory_confidence,
            RuneLiteTab.PRAYER: prayer_confidence,
            RuneLiteTab.EQUIPMENT: equipment_confidence,
            RuneLiteTab.COMBAT: combat_confidence,
            RuneLiteTab.STATS: stats_confidence
        }
        
        best_tab = max(confidence_scores.keys(), key=lambda x: confidence_scores[x])
        best_confidence = confidence_scores[best_tab]
        
        logger.debug(f"🧠 Content analysis: {best_tab.value} ({best_confidence:.2f})")
        logger.debug(f"   Inv={inventory_confidence:.2f}, Combat={combat_confidence:.2f}, Prayer={prayer_confidence:.2f}")
        
        return {
            'tab': best_tab if best_confidence > 0.3 else RuneLiteTab.UNKNOWN,
            'confidence': best_confidence,
            'scores': confidence_scores
        }
    
    def _check_inventory_content(self, screenshot: np.ndarray) -> float:
        """🎒 Check if inventory 4x7 grid is visible"""
        h, w = screenshot.shape[:2]
        
        # CORRECTED inventory area (includes full minimap and tabs)
        # Right side of game viewport, NOT cutting off interface
        inv_x = int(w * 0.82)  # Further right to include full interface
        inv_y = int(h * 0.35)
        inv_w = int(w * 0.13)  # Smaller width since we moved further right
        inv_h = int(h * 0.30)
        
        if inv_x + inv_w > w or inv_y + inv_h > h:
            return 0.0
            
        inv_region = screenshot[inv_y:inv_y+inv_h, inv_x:inv_x+inv_w]
        
        if inv_region.size == 0:
            return 0.0
        
        # Look for 4x7 grid pattern
        gray = cv2.cvtColor(inv_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        
        # Count rectangular regions (inventory slots)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        slot_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 50 < area < 800:  # Slot-sized areas
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.7 < aspect_ratio < 1.5:  # Square-ish (inventory slots)
                    slot_count += 1
        
        # Look for color variety (items have different colors)
        color_variance = np.std(inv_region)
        
        grid_score = min(1.0, slot_count / 15.0)  # Expect many slots
        variety_score = min(1.0, color_variance / 25.0)
        
        confidence = (grid_score + variety_score) / 2.0
        return confidence
    
    def _check_prayer_content(self, screenshot: np.ndarray) -> float:
        """🙏 Check if prayer book is visible"""
        h, w = screenshot.shape[:2]
        
        # Prayer area 
        prayer_x = int(w * 0.82)
        prayer_y = int(h * 0.25)
        prayer_w = int(w * 0.13)
        prayer_h = int(h * 0.40)
        
        if prayer_x + prayer_w > w or prayer_y + prayer_h > h:
            return 0.0
            
        prayer_region = screenshot[prayer_y:prayer_y+prayer_h, prayer_x:prayer_x+prayer_w]
        
        if prayer_region.size == 0:
            return 0.0
        
        # Prayer icons are often circular and have distinctive colors
        gray = cv2.cvtColor(prayer_region, cv2.COLOR_BGR2GRAY)
        
        # Look for circular shapes (prayer icons)
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=15,
            param1=50, param2=30, minRadius=8, maxRadius=25
        )
        
        circle_score = 0.0
        if circles is not None and len(circles[0]) > 0:
            circle_score = min(1.0, len(circles[0]) / 10.0)
        
        # Prayer interface often has bright/glowing effects
        bright_pixels = np.sum(gray > 180)
        brightness_score = min(1.0, (bright_pixels / gray.size) * 5)
        
        confidence = (circle_score + brightness_score) / 2.0
        return confidence
    
    def _check_equipment_content(self, screenshot: np.ndarray) -> float:
        """⚔️ Check if equipment interface (character model) is visible"""
        h, w = screenshot.shape[:2]
        
        equip_x = int(w * 0.82)
        equip_y = int(h * 0.20)
        equip_w = int(w * 0.13)
        equip_h = int(h * 0.45)
        
        if equip_x + equip_w > w or equip_y + equip_h > h:
            return 0.0
            
        equip_region = screenshot[equip_y:equip_y+equip_h, equip_x:equip_x+equip_w]
        
        if equip_region.size == 0:
            return 0.0
        
        # Equipment interface has character model - look for skin tones
        hsv = cv2.cvtColor(equip_region, cv2.COLOR_BGR2HSV)
        
        # Skin tone detection
        lower_skin = np.array([0, 20, 60])
        upper_skin = np.array([25, 255, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        skin_ratio = np.sum(skin_mask > 0) / skin_mask.size
        skin_score = min(1.0, skin_ratio * 15)
        
        # Character model is typically centered
        center_y, center_x = equip_region.shape[0]//2, equip_region.shape[1]//2
        center_region = equip_region[
            center_y-20:center_y+20, 
            center_x-15:center_x+15
        ]
        
        if center_region.size > 0:
            center_variance = np.std(center_region)
            center_score = min(1.0, center_variance / 30.0)
        else:
            center_score = 0.0
        
        confidence = (skin_score + center_score) / 2.0
        return confidence
    
    def _check_combat_content(self, screenshot: np.ndarray) -> float:
        """⚔️ Check if combat options are visible"""
        h, w = screenshot.shape[:2]
        
        combat_x = int(w * 0.82)
        combat_y = int(h * 0.30)
        combat_w = int(w * 0.13)
        combat_h = int(h * 0.35)
        
        if combat_x + combat_w > w or combat_y + combat_h > h:
            return 0.0
            
        combat_region = screenshot[combat_y:combat_y+combat_h, combat_x:combat_x+combat_w]
        
        if combat_region.size == 0:
            return 0.0
        
        # Combat interface has weapon icons and attack style buttons
        gray = cv2.cvtColor(combat_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 120)
        
        # Look for rectangular button-like structures
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        button_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if 100 < area < 1500:  # Button-sized
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 < aspect_ratio < 2.5:  # Button-like
                    button_count += 1
        
        button_score = min(1.0, button_count / 4.0)
        
        # Combat interface has weapon icons (often metallic/distinct colors)
        edge_density = np.sum(edges > 0) / edges.size
        detail_score = min(1.0, edge_density * 8)
        
        confidence = (button_score + detail_score) / 2.0
        return confidence
    
    def _check_stats_content(self, screenshot: np.ndarray) -> float:
        """📊 Check if stats interface is visible"""
        h, w = screenshot.shape[:2]
        
        stats_x = int(w * 0.82)
        stats_y = int(h * 0.25)
        stats_w = int(w * 0.13)
        stats_h = int(h * 0.40)
        
        if stats_x + stats_w > w or stats_y + stats_h > h:
            return 0.0
            
        stats_region = screenshot[stats_y:stats_y+stats_h, stats_x:stats_x+stats_w]
        
        if stats_region.size == 0:
            return 0.0
        
        # Stats has many small skill icons
        gray = cv2.cvtColor(stats_region, cv2.COLOR_BGR2GRAY)
        
        # Look for small circular icons
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, dp=1, minDist=12,
            param1=50, param2=30, minRadius=6, maxRadius=18
        )
        
        icon_score = 0.0
        if circles is not None and len(circles[0]) > 0:
            icon_score = min(1.0, len(circles[0]) / 12.0)  # Stats has many skill icons
        
        # Stats shows numbers (levels) - look for text-like patterns
        edges = cv2.Canny(gray, 30, 90)
        text_density = np.sum(edges > 0) / edges.size
        text_score = min(1.0, text_density * 6)
        
        confidence = (icon_score + text_score) / 2.0
        return confidence
    
    def _detect_orbs_corrected(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """🔮 CORRECTED orb detection positions"""
        orbs = {}
        h, w = screenshot.shape[:2]
        
        # CORRECTED orb positions (in the actual OSRS interface area)
        orb_positions = {
            'health': (0.88, 0.05, 0.05, 0.08),    # Top right of game area
            'prayer': (0.88, 0.15, 0.05, 0.08),   # Below health
            'energy': (0.88, 0.25, 0.05, 0.08),   # Below prayer
            'special': (0.88, 0.35, 0.05, 0.08)   # Below energy
        }
        
        detected_count = 0
        total_confidence = 0.0
        
        for orb_name, (x_ratio, y_ratio, w_ratio, h_ratio) in orb_positions.items():
            orb_x = int(w * x_ratio)
            orb_y = int(h * y_ratio)
            orb_w = int(w * w_ratio)
            orb_h = int(h * h_ratio)
            
            if orb_x + orb_w > w or orb_y + orb_h > h:
                continue
                
            orb_region = screenshot[orb_y:orb_y+orb_h, orb_x:orb_x+orb_w]
            
            if orb_region.size == 0:
                continue
            
            orb_analysis = self._analyze_orb_smart(orb_region, orb_name)
            
            if orb_analysis['confidence'] > 0.3:
                orbs[orb_name] = orb_analysis
                total_confidence += orb_analysis['confidence']
                detected_count += 1
        
        avg_confidence = total_confidence / max(1, detected_count)
        return {'orbs': orbs, 'confidence': avg_confidence}
    
    def _analyze_orb_smart(self, orb_region: np.ndarray, orb_type: str) -> Dict[str, Any]:
        """🔮 Smart orb analysis"""
        try:
            # Basic color analysis
            mean_color = np.mean(orb_region.reshape(-1, 3), axis=0)
            b, g, r = mean_color
            
            # Convert to HSV for better color classification
            bgr_pixel = np.uint8([[[b, g, r]]])
            hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0, 0]
            h, s, v = hsv_pixel
            
            # Classify based on color
            if orb_type == 'health':
                if h < 30 or h > 150:  # Red
                    state = 'low_health'
                elif 30 <= h <= 90:  # Green
                    state = 'high_health'
                else:
                    state = 'medium_health'
            elif orb_type == 'prayer':
                if 90 <= h <= 130:  # Blue
                    state = 'high_prayer'
                else:
                    state = 'low_prayer'
            elif orb_type == 'energy':
                if 15 <= h <= 35:  # Yellow
                    state = 'high_energy'
                else:
                    state = 'low_energy'
            elif orb_type == 'special':
                if 80 <= h <= 100:  # Cyan
                    state = 'special_ready'
                else:
                    state = 'special_charging'
            else:
                state = 'unknown'
            
            # Calculate confidence
            color_variance = np.std(orb_region)
            brightness = np.mean(orb_region)
            
            confidence = min(1.0, (brightness / 255.0) * (color_variance / 30.0))
            
            return {
                'state': state,
                'confidence': max(0.3, confidence),  # Minimum confidence
                'color_info': {'h': int(h), 's': int(s), 'v': int(v)}
            }
            
        except Exception as e:
            return {'state': 'error', 'confidence': 0.0}
    
    def _detect_inventory_smart(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """📦 Smart inventory detection"""
        items = []
        h, w = screenshot.shape[:2]
        
        # CORRECTED inventory position
        inv_x = int(w * 0.82)
        inv_y = int(h * 0.35)
        inv_w = int(w * 0.13)
        inv_h = int(h * 0.30)
        
        if inv_x + inv_w > w or inv_y + inv_h > h:
            return {'items': [], 'confidence': 0.0}
            
        inv_region = screenshot[inv_y:inv_y+inv_h, inv_x:inv_x+inv_w]
        
        if inv_region.size == 0:
            return {'items': [], 'confidence': 0.0}
        
        # 4x7 grid analysis
        grid_w, grid_h = 4, 7
        slot_w = inv_region.shape[1] // grid_w
        slot_h = inv_region.shape[0] // grid_h
        
        for row in range(grid_h):
            for col in range(grid_w):
                x = col * slot_w
                y = row * slot_h
                slot = inv_region[y:y+slot_h, x:x+slot_w]
                
                if slot.size == 0:
                    continue
                
                # Item detection
                color_variance = np.std(slot)
                brightness = np.mean(slot)
                
                # Items have color variation and reasonable brightness
                if color_variance > 8 and brightness > 20:
                    confidence = min(1.0, color_variance / 20.0)
                    
                    items.append({
                        'slot': row * grid_w + col,
                        'name': f'Item_{row}_{col}',
                        'confidence': confidence,
                        'method': 'smart_grid'
                    })
        
        avg_confidence = np.mean([item['confidence'] for item in items]) if items else 0.0
        return {'items': items, 'confidence': avg_confidence}
    
    def _detect_chat_corrected(self, screenshot: np.ndarray) -> Dict[str, Any]:
        """💬 CORRECTED chat detection"""
        messages = []
        h, w = screenshot.shape[:2]
        
        # CORRECTED chat area (bottom of full game viewport)
        chat_x = int(w * 0.01)
        chat_y = int(h * 0.75)
        chat_w = int(w * 0.94)  # Extends to full corrected viewport
        chat_h = int(h * 0.24)
        
        if chat_x + chat_w > w or chat_y + chat_h > h:
            chat_w = w - chat_x
            chat_h = h - chat_y
            
        chat_region = screenshot[chat_y:chat_y+chat_h, chat_x:chat_x+chat_w]
        
        if chat_region.size == 0:
            return {'messages': [], 'confidence': 0.0}
        
        # Text detection
        gray = cv2.cvtColor(chat_region, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            if 15 < w < 500 and 8 < h < 35:
                aspect_ratio = w / h
                if 2 < aspect_ratio < 25:
                    messages.append({
                        'type': 'corrected_chat',
                        'position': (x + w//2, y + h//2),
                        'size': (w, h),
                        'confidence': 0.7
                    })
        
        avg_confidence = 0.7 if messages else 0.0
        return {'messages': messages, 'confidence': avg_confidence}
    
    def _empty_results(self) -> Dict[str, Any]:
        """Empty results"""
        return {
            'active_tab': RuneLiteTab.UNKNOWN,
            'orbs': {},
            'inventory_items': [],
            'chat_messages': [],
            'confidence_scores': {},
            'detection_method': 'ultra_advanced'
        }

# Global instance
ultra_advanced_runelite_vision = UltraAdvancedRuneLiteVision() 