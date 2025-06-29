"""
Global settings and configuration for OSRS Bot Framework
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEMPLATES_DIR = DATA_DIR / "templates"
MODELS_DIR = DATA_DIR / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Ensure directories exist
for directory in [DATA_DIR, TEMPLATES_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Screen capture settings
SCREEN_CAPTURE = {
    "fps": 10,  # Frames per second for continuous capture
    "region": None,  # Will be set during calibration
    "compression": "png",  # Image format for capture
}

# Computer vision settings
VISION = {
    "template_matching": {
        "threshold": 0.8,  # Confidence threshold for template matching
        "method": "TM_CCOEFF_NORMED",  # OpenCV template matching method
        "max_matches": 10,  # Maximum number of matches to find
    },
    "feature_detection": {
        "algorithm": "ORB",  # ORB, SIFT, or SURF
        "max_features": 500,
        "match_threshold": 0.75,
    },
    "color_detection": {
        "hsv_tolerance": 10,  # HSV color matching tolerance
        "min_area": 50,  # Minimum contour area
    },
}

# Alias for backward compatibility
VISION_SETTINGS = VISION

# Automation settings
AUTOMATION = {
    "mouse": {
        "speed": "human",  # "instant", "fast", "human", "slow"
        "randomization": True,  # Add random variations to movements
        "smooth_movement": True,  # Use bezier curves for mouse movement
    },
    "keyboard": {
        "typing_speed": 0.05,  # Delay between keystrokes
        "randomization": True,
    },
    "delays": {
        "min_action_delay": 0.1,  # Minimum delay between actions
        "max_action_delay": 0.3,  # Maximum delay between actions
        "anti_ban_breaks": True,  # Enable anti-ban break system
    },
}

# Safety settings
SAFETY = {
    "failsafe": {
        "enabled": True,
        "key": "ctrl+shift+q",  # Emergency stop key combination
        "mouse_corner": True,  # Stop if mouse moved to corner
    },
    "anti_detection": {
        "randomize_timing": True,
        "human_patterns": True,
        "break_intervals": [300, 600],  # Break every 5-10 minutes
        "long_break_chance": 0.1,  # 10% chance for longer break
    },
    "monitoring": {
        "log_actions": True,
        "screenshot_errors": True,
        "performance_tracking": True,
    },
}

# Logging configuration
LOGGING = {
    "level": "INFO",  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_logging": True,
    "console_logging": True,
    "max_log_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
}

# Client detection settings
CLIENT_DETECTION = {
    "window_title": "RuneLite - Solo Tale",  # Fixed capitalization
    "process_name": "osrs",
    "auto_focus": True,
    "require_focus": True,  # Require client to be in focus
    "client_size": (765, 503),  # Standard OSRS client size (flexible)
    "size_tolerance": 500,  # Allow larger size variation for Runelite
    "auto_detect_size": True,  # Automatically adapt to client size
    "fixed_mode": False,  # Support both fixed and resizable mode
    "scale_factor": 1.0,  # Will be calculated based on actual size
}

# Development settings
DEVELOPMENT = {
    "debug_mode": False,
    "visual_debugging": False,  # Show detection overlays
    "save_debug_images": False,
    "performance_profiling": False,
} 