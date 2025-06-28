"""
Automation utilities for mouse and keyboard control
"""

import time
import random
import math
import pyautogui
import numpy as np
from typing import Tuple, Optional, List
from enum import Enum
import logging

from config.settings import AUTOMATION, SAFETY

logger = logging.getLogger(__name__)

# Disable pyautogui failsafe initially, we'll handle it ourselves
pyautogui.FAILSAFE = SAFETY["failsafe"]["mouse_corner"]


class MouseSpeed(Enum):
    """Mouse movement speed presets"""
    INSTANT = 0
    FAST = 0.1
    HUMAN = 0.3
    SLOW = 0.5


class HumanMouse:
    """Human-like mouse movement and clicking"""
    
    def __init__(self):
        self.last_position = pyautogui.position()
        self.movement_history = []
        
    def _bezier_curve(self, start: Tuple[int, int], end: Tuple[int, int], 
                     control_points: Optional[List[Tuple[int, int]]] = None) -> List[Tuple[int, int]]:
        """Generate bezier curve points for smooth mouse movement"""
        if control_points is None:
            # Generate random control points for natural movement
            mid_x = (start[0] + end[0]) // 2
            mid_y = (start[1] + end[1]) // 2
            
            # Add some randomness to make movement less predictable
            offset_x = random.randint(-50, 50)
            offset_y = random.randint(-50, 50)
            
            control_points = [(mid_x + offset_x, mid_y + offset_y)]
        
        points = []
        num_points = max(10, int(self._distance(start, end) / 10))
        
        for i in range(num_points + 1):
            t = i / num_points
            
            if len(control_points) == 1:
                # Quadratic bezier curve
                cp = control_points[0]
                x = (1-t)**2 * start[0] + 2*(1-t)*t * cp[0] + t**2 * end[0]
                y = (1-t)**2 * start[1] + 2*(1-t)*t * cp[1] + t**2 * end[1]
            else:
                # Linear interpolation fallback
                x = start[0] + t * (end[0] - start[0])
                y = start[1] + t * (end[1] - start[1])
            
            points.append((int(x), int(y)))
        
        return points
    
    def _distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """Calculate distance between two points"""
        return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
    
    def _add_noise(self, point: Tuple[int, int], intensity: float = 1.0) -> Tuple[int, int]:
        """Add small random noise to point for human-like movement"""
        if not AUTOMATION["mouse"]["randomization"]:
            return point
        
        noise_x = random.randint(-int(2 * intensity), int(2 * intensity))
        noise_y = random.randint(-int(2 * intensity), int(2 * intensity))
        
        return (point[0] + noise_x, point[1] + noise_y)
    
    def move_to(self, x: int, y: int, speed: str = "human", 
                smooth: bool = True) -> bool:
        """
        Move mouse to specified coordinates with human-like movement
        """
        try:
            start_pos = pyautogui.position()
            target_pos = (x, y)
            
            # Add small random offset to target
            if AUTOMATION["mouse"]["randomization"]:
                target_pos = self._add_noise(target_pos)
            
            distance = self._distance(start_pos, target_pos)
            
            # Skip movement if already at target
            if distance < 5:
                return True
            
            # Determine movement duration based on speed
            speed_map = {
                "instant": 0,
                "fast": 0.1,
                "human": 0.2 + (distance / 1000),  # Scale with distance
                "slow": 0.5 + (distance / 500)
            }
            
            duration = speed_map.get(speed, speed_map["human"])
            
            if duration == 0 or not smooth:
                # Instant movement
                pyautogui.moveTo(target_pos[0], target_pos[1])
            else:
                # Smooth movement using bezier curve
                if AUTOMATION["mouse"]["smooth_movement"]:
                    curve_points = self._bezier_curve(start_pos, target_pos)
                    
                    for i, point in enumerate(curve_points):
                        # Add micro-pauses and noise
                        if i > 0:
                            micro_delay = duration / len(curve_points)
                            micro_delay += random.uniform(0, micro_delay * 0.3)
                            time.sleep(micro_delay)
                        
                        noisy_point = self._add_noise(point, 0.5)
                        pyautogui.moveTo(noisy_point[0], noisy_point[1])
                else:
                    # Simple linear movement
                    pyautogui.moveTo(target_pos[0], target_pos[1], duration=duration)
            
            # Update tracking
            self.last_position = pyautogui.position()
            self.movement_history.append((time.time(), self.last_position))
            
            # Keep history limited
            if len(self.movement_history) > 100:
                self.movement_history.pop(0)
            
            logger.debug(f"Mouse moved to {self.last_position}")
            return True
            
        except Exception as e:
            logger.error(f"Mouse movement failed: {e}")
            return False
    
    def click(self, x: Optional[int] = None, y: Optional[int] = None, 
              button: str = "left", clicks: int = 1, interval: float = 0.1) -> bool:
        """
        Click at specified coordinates or current position
        """
        try:
            # Move to position if specified
            if x is not None and y is not None:
                if not self.move_to(x, y):
                    return False
            
            # Add random delay before clicking
            if AUTOMATION["mouse"]["randomization"]:
                pre_click_delay = random.uniform(0.05, 0.15)
                time.sleep(pre_click_delay)
            
            # Perform click(s)
            for i in range(clicks):
                if i > 0:
                    time.sleep(interval)
                
                pyautogui.click(button=button)
                
                # Add tiny delay after click
                if AUTOMATION["mouse"]["randomization"]:
                    post_click_delay = random.uniform(0.01, 0.05)
                    time.sleep(post_click_delay)
            
            logger.debug(f"Clicked {button} button {clicks} times at {pyautogui.position()}")
            return True
            
        except Exception as e:
            logger.error(f"Click failed: {e}")
            return False
    
    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, 
             duration: float = 0.5, button: str = "left") -> bool:
        """
        Drag from start to end position
        """
        try:
            # Move to start position
            if not self.move_to(start_x, start_y):
                return False
            
            # Perform drag
            pyautogui.drag(end_x - start_x, end_y - start_y, 
                          duration=duration, button=button)
            
            logger.debug(f"Dragged from ({start_x}, {start_y}) to ({end_x}, {end_y})")
            return True
            
        except Exception as e:
            logger.error(f"Drag failed: {e}")
            return False
    
    def scroll(self, clicks: int, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """
        Scroll at specified position or current position
        """
        try:
            if x is not None and y is not None:
                if not self.move_to(x, y):
                    return False
            
            pyautogui.scroll(clicks)
            logger.debug(f"Scrolled {clicks} clicks at {pyautogui.position()}")
            return True
            
        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            return False


class HumanKeyboard:
    """Human-like keyboard input"""
    
    def __init__(self):
        self.typing_speed = AUTOMATION["keyboard"]["typing_speed"]
        self.randomization = AUTOMATION["keyboard"]["randomization"]
    
    def type_text(self, text: str, speed: Optional[float] = None) -> bool:
        """
        Type text with human-like timing
        """
        try:
            typing_speed = speed or self.typing_speed
            
            for char in text:
                pyautogui.write(char)
                
                if self.randomization:
                    # Add random variation to typing speed
                    delay = typing_speed + random.uniform(-typing_speed * 0.3, 
                                                         typing_speed * 0.3)
                    delay = max(0.01, delay)  # Minimum delay
                    time.sleep(delay)
                else:
                    time.sleep(typing_speed)
            
            logger.debug(f"Typed text: {text[:20]}...")
            return True
            
        except Exception as e:
            logger.error(f"Text typing failed: {e}")
            return False
    
    def press_key(self, key: str, presses: int = 1, interval: float = 0.1) -> bool:
        """
        Press key(s) with optional repetition
        """
        try:
            for i in range(presses):
                if i > 0:
                    time.sleep(interval)
                
                pyautogui.press(key)
                
                if self.randomization:
                    micro_delay = random.uniform(0.01, 0.05)
                    time.sleep(micro_delay)
            
            logger.debug(f"Pressed key '{key}' {presses} times")
            return True
            
        except Exception as e:
            logger.error(f"Key press failed: {e}")
            return False
    
    def key_combination(self, keys: List[str]) -> bool:
        """
        Press key combination (e.g., ['ctrl', 'c'])
        """
        try:
            pyautogui.hotkey(*keys)
            logger.debug(f"Pressed key combination: {'+'.join(keys)}")
            return True
            
        except Exception as e:
            logger.error(f"Key combination failed: {e}")
            return False


# Global instances
mouse = HumanMouse()
keyboard = HumanKeyboard()


def random_delay(min_delay: float = None, max_delay: float = None) -> None:
    """Add random delay between actions"""
    if min_delay is None:
        min_delay = AUTOMATION["delays"]["min_action_delay"]
    if max_delay is None:
        max_delay = AUTOMATION["delays"]["max_action_delay"]
    
    delay = random.uniform(min_delay, max_delay)
    time.sleep(delay)


def emergency_stop() -> None:
    """Emergency stop function"""
    logger.critical("EMERGENCY STOP ACTIVATED")
    # Move mouse to corner to trigger pyautogui failsafe
    if SAFETY["failsafe"]["mouse_corner"]:
        pyautogui.moveTo(0, 0)
    raise KeyboardInterrupt("Emergency stop activated") 