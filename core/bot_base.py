"""
Base bot class for OSRS Bot Framework
"""

import time
import threading
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Callable
from enum import Enum
import logging

from core.screen_capture import screen_capture
from core.automation import mouse, keyboard, random_delay, emergency_stop
from core.computer_vision import cv_system, Detection
from config.settings import SAFETY, AUTOMATION


logger = logging.getLogger(__name__)


class BotState(Enum):
    """Bot execution states"""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class BotBase(ABC):
    """
    Abstract base class for all OSRS bots
    
    Provides common functionality like:
    - State management
    - Screen capture integration
    - Safety mechanisms
    - Logging and monitoring
    - Configuration management
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.state = BotState.IDLE
        
        # Runtime tracking
        self.start_time: Optional[float] = None
        self.total_runtime = 0.0
        self.action_count = 0
        self.error_count = 0
        
        # Break system
        self.last_break_time = time.time()
        self.break_intervals = SAFETY["anti_detection"]["break_intervals"]
        self.long_break_chance = SAFETY["anti_detection"]["long_break_chance"]
        
        # Threading
        self.main_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Callbacks
        self.state_callbacks: Dict[BotState, List[Callable]] = {}
        
        # Performance monitoring
        self.performance_stats = {
            "actions_per_minute": 0.0,
            "avg_cycle_time": 0.0,
            "success_rate": 0.0,
            "errors_per_hour": 0.0
        }
        
        logger.info(f"Bot '{name}' initialized")
    
    def set_state(self, new_state: BotState) -> None:
        """Update bot state and trigger callbacks"""
        old_state = self.state
        self.state = new_state
        
        logger.info(f"Bot '{self.name}' state changed: {old_state.value} -> {new_state.value}")
        
        # Trigger state callbacks
        if new_state in self.state_callbacks:
            for callback in self.state_callbacks[new_state]:
                try:
                    callback(old_state, new_state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")
    
    def add_state_callback(self, state: BotState, callback: Callable) -> None:
        """Add callback for state changes"""
        if state not in self.state_callbacks:
            self.state_callbacks[state] = []
        self.state_callbacks[state].append(callback)
    
    def start(self, threaded: bool = True) -> bool:
        """
        Start the bot
        
        Args:
            threaded: If True, run bot in separate thread
        
        Returns:
            True if started successfully
        """
        if self.state not in [BotState.IDLE, BotState.STOPPED]:
            logger.warning(f"Cannot start bot in state: {self.state.value}")
            return False
        
        try:
            self.set_state(BotState.INITIALIZING)
            
            # Pre-flight checks
            if not self._pre_flight_checks():
                self.set_state(BotState.ERROR)
                return False
            
            # Initialize bot-specific components
            if not self.initialize():
                self.set_state(BotState.ERROR)
                return False
            
            # Start execution
            self.start_time = time.time()
            self.stop_event.clear()
            
            if threaded:
                self.main_thread = threading.Thread(target=self._run_loop, daemon=True)
                self.main_thread.start()
            else:
                self._run_loop()
            
            self.set_state(BotState.RUNNING)
            logger.info(f"Bot '{self.name}' started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start bot '{self.name}': {e}")
            self.set_state(BotState.ERROR)
            return False
    
    def stop(self, timeout: float = 5.0) -> bool:
        """
        Stop the bot gracefully
        
        Args:
            timeout: Maximum time to wait for graceful shutdown
        
        Returns:
            True if stopped successfully
        """
        if self.state in [BotState.STOPPED, BotState.STOPPING]:
            return True
        
        logger.info(f"Stopping bot '{self.name}'...")
        self.set_state(BotState.STOPPING)
        
        # Signal stop
        self.stop_event.set()
        
        # Wait for thread to finish
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=timeout)
            
            if self.main_thread.is_alive():
                logger.warning(f"Bot '{self.name}' did not stop gracefully within {timeout}s")
                return False
        
        # Update runtime stats
        if self.start_time:
            self.total_runtime += time.time() - self.start_time
            self.start_time = None
        
        self.set_state(BotState.STOPPED)
        logger.info(f"Bot '{self.name}' stopped")
        return True
    
    def pause(self) -> bool:
        """Pause the bot"""
        if self.state != BotState.RUNNING:
            return False
        
        self.set_state(BotState.PAUSED)
        logger.info(f"Bot '{self.name}' paused")
        return True
    
    def resume(self) -> bool:
        """Resume the bot from pause"""
        if self.state != BotState.PAUSED:
            return False
        
        self.set_state(BotState.RUNNING)
        logger.info(f"Bot '{self.name}' resumed")
        return True
    
    def emergency_stop(self) -> None:
        """Emergency stop with immediate termination"""
        logger.critical(f"EMERGENCY STOP - Bot '{self.name}'")
        self.stop_event.set()
        self.set_state(BotState.STOPPED)
        emergency_stop()
    
    def _pre_flight_checks(self) -> bool:
        """Perform pre-flight safety and setup checks"""
        try:
            # Check if client is available
            if not screen_capture.calibrate_client():
                logger.error("OSRS client not found or not accessible")
                return False
            
            # Verify client is active if required
            if self.config.get("require_focus", True):
                if not screen_capture.is_client_active():
                    logger.error("OSRS client is not active/focused")
                    return False
            
            # Check safety systems
            if SAFETY["failsafe"]["enabled"]:
                logger.info("Safety failsafe enabled")
            
            return True
            
        except Exception as e:
            logger.error(f"Pre-flight checks failed: {e}")
            return False
    
    def _run_loop(self) -> None:
        """Main execution loop"""
        try:
            cycle_times = []
            
            while not self.stop_event.is_set():
                cycle_start = time.time()
                
                # Handle paused state
                if self.state == BotState.PAUSED:
                    time.sleep(0.1)
                    continue
                
                # Check for breaks
                if self._should_take_break():
                    self._take_break()
                    continue
                
                # Execute main bot logic
                try:
                    success = self.execute_cycle()
                    
                    if success:
                        self.action_count += 1
                    else:
                        self.error_count += 1
                        logger.warning(f"Bot cycle failed for '{self.name}'")
                    
                except Exception as e:
                    self.error_count += 1
                    logger.error(f"Bot cycle error for '{self.name}': {e}")
                    
                    # Stop on critical errors
                    if self.error_count > 10:
                        logger.critical(f"Too many errors, stopping bot '{self.name}'")
                        break
                
                # Track performance
                cycle_time = time.time() - cycle_start
                cycle_times.append(cycle_time)
                if len(cycle_times) > 100:
                    cycle_times.pop(0)
                
                # Update performance stats
                self._update_performance_stats(cycle_times)
                
                # Anti-detection delay
                if AUTOMATION["delays"]["anti_ban_breaks"]:
                    random_delay()
            
        except KeyboardInterrupt:
            logger.info(f"Bot '{self.name}' interrupted by user")
        except Exception as e:
            logger.error(f"Bot '{self.name}' crashed: {e}")
            self.set_state(BotState.ERROR)
        finally:
            self.cleanup()
    
    def _should_take_break(self) -> bool:
        """Check if bot should take a break"""
        if not SAFETY["anti_detection"]["randomize_timing"]:
            return False
        
        time_since_break = time.time() - self.last_break_time
        min_interval, max_interval = self.break_intervals
        
        # Random break interval
        import random
        break_interval = random.uniform(min_interval, max_interval)
        
        return time_since_break >= break_interval
    
    def _take_break(self) -> None:
        """Take an anti-detection break"""
        import random
        
        # Determine break duration
        base_break = random.uniform(10, 30)  # 10-30 seconds base
        
        # Chance for longer break
        if random.random() < self.long_break_chance:
            base_break *= random.uniform(5, 15)  # 5-15x longer
        
        logger.info(f"Bot '{self.name}' taking break for {base_break:.1f} seconds")
        
        self.last_break_time = time.time()
        time.sleep(base_break)
    
    def _update_performance_stats(self, cycle_times: List[float]) -> None:
        """Update performance statistics"""
        if not cycle_times:
            return
        
        runtime = time.time() - (self.start_time or time.time())
        
        self.performance_stats.update({
            "actions_per_minute": (self.action_count / runtime) * 60 if runtime > 0 else 0,
            "avg_cycle_time": sum(cycle_times) / len(cycle_times),
            "success_rate": (self.action_count / (self.action_count + self.error_count)) * 100 
                           if (self.action_count + self.error_count) > 0 else 0,
            "errors_per_hour": (self.error_count / runtime) * 3600 if runtime > 0 else 0
        })
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive bot status"""
        runtime = 0
        if self.start_time:
            runtime = time.time() - self.start_time
        
        return {
            "name": self.name,
            "state": self.state.value,
            "runtime": runtime + self.total_runtime,
            "actions": self.action_count,
            "errors": self.error_count,
            "performance": self.performance_stats.copy(),
            "config": self.config.copy()
        }
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize bot-specific components
        Called once before bot starts running
        
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def execute_cycle(self) -> bool:
        """
        Execute one cycle of bot logic
        Called repeatedly while bot is running
        
        Returns:
            True if cycle completed successfully
        """
        pass
    
    def cleanup(self) -> None:
        """
        Clean up resources when bot stops
        Override in subclass if needed
        """
        pass
    
    # Utility methods for common bot operations
    
    def capture_and_process(self) -> Optional[Dict[str, Any]]:
        """Capture screen and process with computer vision"""
        try:
            image = screen_capture.capture_client()
            if image is None:
                return None
            
            return cv_system.process_image(image)
            
        except Exception as e:
            logger.error(f"Failed to capture and process image: {e}")
            return None
    
    def find_and_click(self, template_name: str, confidence: float = 0.8,
                      timeout: float = 5.0) -> bool:
        """
        Find template and click on it
        
        Args:
            template_name: Name of template to find
            confidence: Minimum confidence threshold
            timeout: Maximum time to search
        
        Returns:
            True if found and clicked successfully
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.stop_event.is_set():
                return False
            
            image = screen_capture.capture_client()
            if image is None:
                continue
            
            detections = cv_system.template_matching.find_template(
                image, template_name, threshold=confidence
            )
            
            if detections:
                # Click on the first (best) match
                detection = detections[0]
                center_x, center_y = detection.center
                
                # Convert to screen coordinates
                if screen_capture.client_region:
                    screen_x = screen_capture.client_region.x + center_x
                    screen_y = screen_capture.client_region.y + center_y
                    
                    if mouse.click(screen_x, screen_y):
                        logger.info(f"Clicked on '{template_name}' at ({screen_x}, {screen_y})")
                        return True
            
            time.sleep(0.1)  # Short delay between attempts
        
        logger.warning(f"Template '{template_name}' not found within {timeout}s")
        return False
    
    def wait_for_template(self, template_name: str, confidence: float = 0.8,
                         timeout: float = 10.0) -> bool:
        """
        Wait for template to appear
        
        Returns:
            True if template appeared within timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.stop_event.is_set():
                return False
            
            image = screen_capture.capture_client()
            if image is None:
                continue
            
            detections = cv_system.template_matching.find_template(
                image, template_name, threshold=confidence
            )
            
            if detections:
                return True
            
            time.sleep(0.1)
        
        return False 