"""
Screen capture utilities for OSRS Bot Framework
"""

import time
import cv2
import numpy as np
import mss
import pyautogui
import threading
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import logging

from config.settings import SCREEN_CAPTURE, CLIENT_DETECTION, DEVELOPMENT

logger = logging.getLogger(__name__)

# Import adaptive vision
try:
    from core.adaptive_vision import adaptive_vision
except ImportError:
    adaptive_vision = None


@dataclass
class ClientRegion:
    """Represents the OSRS client region on screen"""
    x: int
    y: int
    width: int
    height: int
    
    @property
    def bbox(self) -> Dict[str, int]:
        """Return bounding box for mss capture"""
        return {
            'top': self.y,
            'left': self.x,
            'width': self.width,
            'height': self.height
        }
    
    @property
    def center(self) -> Tuple[int, int]:
        """Return center coordinates of the client"""
        return (self.x + self.width // 2, self.y + self.height // 2)


class ScreenCapture:
    """High-performance screen capture for OSRS client"""
    
    def __init__(self):
        # Use thread-local storage for MSS instances to prevent threading issues
        self._local = threading.local()
        self.client_region: Optional[ClientRegion] = None
        self.last_capture_time = 0
        self.fps_limit = SCREEN_CAPTURE["fps"]
        self.min_frame_time = 1.0 / self.fps_limit if self.fps_limit > 0 else 0
        
        # Performance tracking
        self.capture_times = []
        self.frame_count = 0
        
        # Warning tracking to avoid spam
        self._large_client_warned = False
    
    @property
    def sct(self):
        """Get thread-local MSS instance"""
        if not hasattr(self._local, 'sct'):
            self._local.sct = mss.mss()
        return self._local.sct
        
    def find_client_window(self) -> Optional[ClientRegion]:
        """
        Find and return the OSRS client window region
        Returns None if client not found
        """
        try:
            # Try to find window by title
            windows = pyautogui.getWindowsWithTitle(CLIENT_DETECTION["window_title"])
            
            if not windows:
                logger.warning(f"No window found with title: {CLIENT_DETECTION['window_title']}")
                return None
            
            # Use the first matching window
            window = windows[0]
            
            # Get window position and size
            region = ClientRegion(
                x=window.left,
                y=window.top,
                width=window.width,
                height=window.height
            )
            
            # Validate window size (if auto-detect is disabled)
            if not CLIENT_DETECTION.get("auto_detect_size", True):
                expected_size = CLIENT_DETECTION["client_size"]
                tolerance = CLIENT_DETECTION.get("size_tolerance", 50)
                if (abs(region.width - expected_size[0]) > tolerance or 
                    abs(region.height - expected_size[1]) > tolerance):
                    logger.warning(f"Client size mismatch. Expected: {expected_size}, "
                                 f"Found: {region.width}x{region.height}")
            else:
                # Auto-detect and adapt to current client size
                logger.info(f"Auto-detected client size: {region.width}x{region.height}")
                
                # Configure adaptive vision
                if adaptive_vision:
                    adaptive_vision.set_client_size(region.width, region.height)
                    impact = adaptive_vision.get_performance_impact()
                    logger.info(f"Performance impact: {impact}")
                    
                    if adaptive_vision.is_large_client() and not self._large_client_warned:
                        logger.warning("Large client detected - consider smaller window for better performance")
                        recommendations = adaptive_vision.get_recommendations()
                        for rec in recommendations:
                            logger.info(f"💡 Recommendation: {rec}")
                        self._large_client_warned = True
                else:
                    if (region.width > 1000 or region.height > 600) and not self._large_client_warned:
                        logger.info("Large client detected - this may impact performance but will work")
                        self._large_client_warned = True
            
            logger.info(f"Found OSRS client at {region.x}, {region.y} "
                       f"({region.width}x{region.height})")
            
            return region
            
        except Exception as e:
            logger.error(f"Error finding client window: {e}")
            return None
    
    def calibrate_client(self) -> bool:
        """
        Calibrate the screen capture to the OSRS client
        Returns True if successful
        """
        logger.info("Calibrating client region...")
        
        try:
            self.client_region = self.find_client_window()
            
            if self.client_region is None:
                logger.error(f"Failed to find OSRS client window with title: '{CLIENT_DETECTION['window_title']}'")
                logger.info("Make sure RuneLite is open and not minimized")
                return False
            
            logger.info(f"Found client at ({self.client_region.x}, {self.client_region.y}) "
                       f"size {self.client_region.width}x{self.client_region.height}")
            
            # Test capture with the thread-safe MSS instance
            test_image = self.capture_client()
            if test_image is None:
                logger.error("Failed to capture test image - screen capture not working")
                return False
            
            logger.info(f"Client calibration successful! Screenshot: {test_image.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error during client calibration: {e}")
            return False
    
    def capture_screen(self, region: Optional[Dict[str, int]] = None) -> Optional[np.ndarray]:
        """
        Capture full screen or specified region
        Returns BGR image as numpy array
        """
        try:
            # Throttle capture rate if FPS limit set
            if self.min_frame_time > 0:
                current_time = time.time()
                time_since_last = current_time - self.last_capture_time
                if time_since_last < self.min_frame_time:
                    time.sleep(self.min_frame_time - time_since_last)
                self.last_capture_time = time.time()
            
            # Capture screen
            start_time = time.time()
            
            try:
                if region:
                    logger.debug(f"Capturing region: {region}")
                    screenshot = self.sct.grab(region)
                else:
                    screenshot = self.sct.grab(self.sct.monitors[1])  # Primary monitor
            except Exception as grab_error:
                logger.error(f"MSS grab failed: {grab_error}")
                # Try creating a new MSS instance
                try:
                    logger.info("Attempting to reinitialize MSS...")
                    self._local.sct = mss.mss()
                    if region:
                        screenshot = self.sct.grab(region)
                    else:
                        screenshot = self.sct.grab(self.sct.monitors[1])
                    logger.info("MSS reinitialization successful")
                except Exception as retry_error:
                    logger.error(f"MSS reinitialization failed: {retry_error}")
                    return None
            
            # Convert to numpy array (BGR format)
            image = np.array(screenshot)
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            
            # Track performance
            capture_time = time.time() - start_time
            self.capture_times.append(capture_time)
            if len(self.capture_times) > 100:
                self.capture_times.pop(0)
            
            self.frame_count += 1
            
            return image
            
        except Exception as e:
            logger.error(f"Screen capture failed: {e}")
            return None
    
    def capture_client(self) -> Optional[np.ndarray]:
        """
        Capture only the OSRS client region
        Returns BGR image as numpy array
        """
        if self.client_region is None:
            logger.warning("Client region not calibrated")
            return None
        
        return self.capture_screen(self.client_region.bbox)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics for screen capture"""
        if not self.capture_times:
            return {}
        
        avg_time = sum(self.capture_times) / len(self.capture_times)
        max_time = max(self.capture_times)
        min_time = min(self.capture_times)
        
        return {
            "avg_capture_time": avg_time,
            "max_capture_time": max_time,
            "min_capture_time": min_time,
            "avg_fps": 1.0 / avg_time if avg_time > 0 else 0,
            "total_frames": self.frame_count
        }
    
    def save_debug_screenshot(self, image: np.ndarray, filename: str) -> bool:
        """Save image for debugging purposes"""
        if not DEVELOPMENT["save_debug_images"]:
            return False
        
        try:
            from pathlib import Path
            debug_dir = Path("debug_images")
            debug_dir.mkdir(exist_ok=True)
            
            filepath = debug_dir / f"{filename}_{int(time.time())}.png"
            cv2.imwrite(str(filepath), image)
            logger.debug(f"Debug screenshot saved: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save debug screenshot: {e}")
            return False
    
    def is_client_active(self) -> bool:
        """Check if OSRS client is active/focused"""
        try:
            active_window = pyautogui.getActiveWindow()
            if active_window is None:
                return False
            
            return CLIENT_DETECTION["window_title"] in active_window.title
            
        except Exception as e:
            logger.debug(f"Error checking active window: {e}")
            return False
    
    def focus_client(self) -> bool:
        """Bring OSRS client to focus"""
        try:
            windows = pyautogui.getWindowsWithTitle(CLIENT_DETECTION["window_title"])
            if not windows:
                return False
            
            window = windows[0]
            window.activate()
            time.sleep(0.1)  # Give time for window to focus
            
            return self.is_client_active()
            
        except Exception as e:
            logger.error(f"Failed to focus client: {e}")
            return False


# Global screen capture instance
screen_capture = ScreenCapture() 