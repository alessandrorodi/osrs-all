#!/usr/bin/env python3
"""
OSRS Bot Framework Setup Script

This script performs initial setup and configuration.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.logging import setup_logging
from core.screen_capture import screen_capture
from config.settings import TEMPLATES_DIR, LOGS_DIR, DATA_DIR, MODELS_DIR

logger = setup_logging("setup")


def create_directories():
    """Create necessary directories"""
    directories = [
        TEMPLATES_DIR,
        LOGS_DIR,
        DATA_DIR,
        MODELS_DIR,
        project_root / "debug_images",
        project_root / "vision" / "detectors",
        project_root / "bots",
        project_root / "tools"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def test_dependencies():
    """Test if all required dependencies are available"""
    logger.info("Testing dependencies...")
    
    dependencies = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("mss", "MSS Screen Capture"),
        ("pyautogui", "PyAutoGUI"),
        ("pynput", "PyNput"),
        ("yaml", "PyYAML"),
        ("colorlog", "ColorLog"),
        ("click", "Click"),
        ("sklearn", "Scikit-learn")
    ]
    
    missing = []
    for module, name in dependencies:
        try:
            __import__(module)
            logger.info(f"✓ {name}")
        except ImportError:
            logger.error(f"✗ {name} - MISSING")
            missing.append(name)
    
    if missing:
        logger.error(f"Missing dependencies: {', '.join(missing)}")
        logger.error("Please install missing dependencies with: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are available!")
    return True


def test_screen_capture():
    """Test screen capture functionality"""
    logger.info("Testing screen capture...")
    
    try:
        # Test basic screen capture
        image = screen_capture.capture_screen()
        if image is not None:
            logger.info(f"✓ Screen capture working (captured {image.shape[1]}x{image.shape[0]} image)")
        else:
            logger.error("✗ Screen capture failed")
            return False
        
        # Test client detection
        client_region = screen_capture.find_client_window()
        if client_region:
            logger.info(f"✓ OSRS client detected at {client_region.x}, {client_region.y} ({client_region.width}x{client_region.height})")
        else:
            logger.warning("⚠ OSRS client not found (this is normal if client is not running)")
        
        return True
        
    except Exception as e:
        logger.error(f"Screen capture test failed: {e}")
        return False


def test_automation():
    """Test automation capabilities"""
    logger.info("Testing automation...")
    
    try:
        from core.automation import mouse, keyboard
        
        # Test mouse position detection
        pos = mouse.last_position
        logger.info(f"✓ Mouse position: {pos}")
        
        # Test keyboard (non-destructive)
        logger.info("✓ Keyboard automation ready")
        
        return True
        
    except Exception as e:
        logger.error(f"Automation test failed: {e}")
        return False


def test_computer_vision():
    """Test computer vision system"""
    logger.info("Testing computer vision...")
    
    try:
        from core.computer_vision import cv_system
        
        logger.info(f"✓ Template manager loaded {len(cv_system.template_manager.templates)} templates")
        logger.info("✓ Computer vision system ready")
        
        return True
        
    except Exception as e:
        logger.error(f"Computer vision test failed: {e}")
        return False


def create_example_config():
    """Create example configuration files"""
    logger.info("Creating example configuration files...")
    
    # Example bot config
    example_config = """# Example Bot Configuration
name: "example_bot"
enabled: true

# Bot-specific settings
settings:
  target_monster: "goblin"
  combat_style: "melee"
  food_threshold: 50
  
# Detection templates
templates:
  - "goblin"
  - "health_bar"
  - "food_item"

# Regions of interest (relative to client)
regions:
  inventory: [548, 205, 190, 261]
  minimap: [570, 9, 146, 151]
  chat: [7, 345, 506, 120]
"""
    
    config_file = project_root / "config" / "example_bot.yaml"
    with open(config_file, 'w') as f:
        f.write(example_config)
    
    logger.info(f"Created example config: {config_file}")


def main():
    """Main setup function"""
    logger.info("OSRS Bot Framework Setup")
    logger.info("=" * 40)
    
    # Create directories
    create_directories()
    
    # Test dependencies
    if not test_dependencies():
        return False
    
    # Test core components
    tests = [
        test_screen_capture,
        test_automation,
        test_computer_vision
    ]
    
    for test in tests:
        if not test():
            logger.error("Setup failed!")
            return False
    
    # Create example files
    create_example_config()
    
    logger.info("=" * 40)
    logger.info("Setup completed successfully!")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Run 'python tools/calibrate_client.py' to calibrate your OSRS client")
    logger.info("2. Use 'python tools/template_creator.py' to create detection templates")
    logger.info("3. Create your first bot by extending the BotBase class")
    logger.info("")
    logger.info("For help and documentation, see README.md")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 