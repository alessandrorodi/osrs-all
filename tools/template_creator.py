#!/usr/bin/env python3
"""
Template Creator Tool

Interactive tool for creating template images for object detection.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from utils.logging import setup_logging
from core.screen_capture import screen_capture
from core.computer_vision import cv_system

logger = setup_logging("template_creator")

# Global variables for mouse callback
drawing = False
start_x, start_y = -1, -1
current_selection = None


def mouse_callback(event, x, y, flags, param):
    """Mouse callback for selecting regions"""
    global drawing, start_x, start_y, current_selection
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        current_selection = None
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_selection = (min(start_x, x), min(start_y, y), 
                               abs(x - start_x), abs(y - start_y))
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if abs(x - start_x) > 10 and abs(y - start_y) > 10:
            current_selection = (min(start_x, x), min(start_y, y), 
                               abs(x - start_x), abs(y - start_y))


def capture_and_save_template():
    """Interactive template capture session"""
    logger.info("Starting template capture session...")
    
    if not screen_capture.calibrate_client():
        logger.error("Could not calibrate client. Make sure OSRS is running.")
        return False
    
    print("\nTemplate Creator Instructions:")
    print("- A window will show your OSRS client")
    print("- Click and drag to select objects/UI elements")
    print("- Press 's' to save the selected region as a template")
    print("- Press 'c' to capture a new screenshot")
    print("- Press 'q' to quit")
    print("\nPress Enter to start...")
    input()
    
    # Create window
    cv2.namedWindow("Template Creator", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Template Creator", mouse_callback)
    
    # Initial capture
    image = screen_capture.capture_client()
    if image is None:
        logger.error("Could not capture client image")
        return False
    
    display_image = image.copy()
    
    while True:
        # Draw current selection
        temp_image = display_image.copy()
        if current_selection:
            x, y, w, h = current_selection
            cv2.rectangle(temp_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(temp_image, f"Selection: {w}x{h}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow("Template Creator", temp_image)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Capture new screenshot
            logger.info("Capturing new screenshot...")
            new_image = screen_capture.capture_client()
            if new_image is not None:
                image = new_image
                display_image = image.copy()
                current_selection = None
                logger.info("New screenshot captured")
        elif key == ord('s') and current_selection:
            # Save template
            x, y, w, h = current_selection
            template = image[y:y+h, x:x+w]
            
            print(f"\nSelected region: {w}x{h} at ({x}, {y})")
            template_name = input("Enter template name (or press Enter to skip): ").strip()
            
            if template_name:
                if cv_system.template_manager.save_template(template_name, template):
                    print(f"✓ Template '{template_name}' saved successfully!")
                    logger.info(f"Saved template: {template_name} ({w}x{h})")
                else:
                    print(f"✗ Failed to save template '{template_name}'")
            
            current_selection = None
    
    cv2.destroyAllWindows()
    logger.info("Template creation session ended")
    return True


def list_existing_templates():
    """List all existing templates"""
    templates = cv_system.template_manager.templates
    
    if not templates:
        print("No templates found.")
        return
    
    print(f"\nExisting templates ({len(templates)}):")
    print("-" * 40)
    
    for name, template in templates.items():
        size = cv_system.template_manager.template_sizes.get(name, (0, 0))
        print(f"  {name}: {size[0]}x{size[1]} pixels")


def test_template_matching():
    """Test template matching with current client capture"""
    templates = cv_system.template_manager.templates
    
    if not templates:
        print("No templates available for testing.")
        return
    
    if not screen_capture.calibrate_client():
        logger.error("Could not calibrate client")
        return
    
    print("\nTesting template matching...")
    print("Available templates:")
    template_names = list(templates.keys())
    for i, name in enumerate(template_names):
        print(f"  {i + 1}. {name}")
    
    try:
        choice = input("\nEnter template number to test (or 'all' for all): ").strip()
        
        image = screen_capture.capture_client()
        if image is None:
            logger.error("Could not capture client image")
            return
        
        if choice.lower() == 'all':
            test_templates = template_names
        else:
            index = int(choice) - 1
            if 0 <= index < len(template_names):
                test_templates = [template_names[index]]
            else:
                print("Invalid choice")
                return
        
        # Test matching
        for template_name in test_templates:
            detections = cv_system.template_matching.find_template(image, template_name)
            
            print(f"\nTemplate '{template_name}':")
            if detections:
                for i, det in enumerate(detections):
                    print(f"  Match {i + 1}: confidence={det.confidence:.3f}, "
                          f"position=({det.x}, {det.y}), size=({det.width}x{det.height})")
            else:
                print("  No matches found")
        
        # Visual debug
        if input("\nShow visual results? (y/n): ").lower() == 'y':
            all_detections = []
            for template_name in test_templates:
                detections = cv_system.template_matching.find_template(image, template_name)
                all_detections.extend(detections)
            
            debug_image = cv_system.debug_visualize(image, all_detections, "Template Test")
            cv2.imshow("Template Test Results", debug_image)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    except ValueError:
        print("Invalid input")
    except Exception as e:
        logger.error(f"Template testing failed: {e}")


def main():
    """Main function"""
    logger.info("OSRS Template Creator Tool")
    print("=" * 40)
    
    while True:
        print("\nOptions:")
        print("1. Create new templates")
        print("2. List existing templates")
        print("3. Test template matching")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            capture_and_save_template()
        elif choice == '2':
            list_existing_templates()
        elif choice == '3':
            test_template_matching()
        elif choice == '4':
            break
        else:
            print("Invalid choice")
    
    logger.info("Template creator tool finished")


if __name__ == "__main__":
    main() 