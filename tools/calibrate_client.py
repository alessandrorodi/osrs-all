#!/usr/bin/env python3
"""
Client Calibration Tool

This tool helps calibrate the screen capture system for your OSRS client.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
from utils.logging import setup_logging
from core.screen_capture import screen_capture
from config.settings import CLIENT_DETECTION

logger = setup_logging("calibrate_client")


def main():
    """Main calibration function"""
    logger.info("OSRS Client Calibration Tool")
    logger.info("=" * 40)
    
    print("\nInstructions:")
    print("1. Make sure your Old School RuneScape client is open")
    print("2. Position it where you want it during botting")
    print("3. Press Enter to start calibration...")
    
    input()
    
    # Find and calibrate client
    logger.info("Searching for OSRS client...")
    
    if screen_capture.calibrate_client():
        region = screen_capture.client_region
        logger.info(f"✓ Client found and calibrated!")
        logger.info(f"  Position: ({region.x}, {region.y})")
        logger.info(f"  Size: {region.width}x{region.height}")
        
        # Test capture
        logger.info("Testing screen capture...")
        image = screen_capture.capture_client()
        
        if image is not None:
            logger.info(f"✓ Screen capture test successful!")
            logger.info(f"  Captured image: {image.shape[1]}x{image.shape[0]} pixels")
            
            # Save test capture
            debug_path = project_root / "debug_images"
            debug_path.mkdir(exist_ok=True)
            
            test_file = debug_path / "calibration_test.png"
            cv2.imwrite(str(test_file), image)
            logger.info(f"  Test image saved: {test_file}")
            
            # Show performance stats
            stats = screen_capture.get_performance_stats()
            if stats:
                logger.info(f"  Capture performance: {stats['avg_fps']:.1f} FPS")
            
            print(f"\n✓ Calibration completed successfully!")
            print(f"Client region: {region.x}, {region.y}, {region.width}x{region.height}")
            print(f"Test image saved to: {test_file}")
            
        else:
            logger.error("✗ Screen capture test failed!")
            return False
            
    else:
        logger.error("✗ Could not find OSRS client!")
        print(f"\nTroubleshooting:")
        print(f"- Make sure OSRS client is open")
        print(f"- Check that window title contains: '{CLIENT_DETECTION['window_title']}'")
        print(f"- Try running as administrator")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 