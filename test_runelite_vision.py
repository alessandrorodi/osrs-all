#!/usr/bin/env python3
"""
🚀 ADVANCED RuneLite Vision System Test Suite
Tests cutting-edge computer vision capabilities with live RuneLite client
Uses machine learning, template matching, and advanced image analysis
"""

import sys
import cv2
import numpy as np
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.screen_capture import ScreenCapture
from vision.ultra_advanced_vision import ultra_advanced_runelite_vision, RuneLiteTab
from utils.logging import get_logger

logger = get_logger(__name__)

def test_runelite_detection():
    """Test RuneLite vision system with live screenshots"""
    print("🚀🧠 Testing ULTRA-ADVANCED RuneLite Vision System")
    print("✅ FIXED: Viewport Detection + Content-Aware Tab Detection")
    print("=" * 65)
    
    # Initialize screen capture
    screen_capture = ScreenCapture()
    
    # Calibrate to find RuneLite client
    print("🎯 Calibrating RuneLite client...")
    if not screen_capture.calibrate_client():
        print("❌ Failed to find RuneLite client!")
        print("   Make sure RuneLite is open and not minimized")
        return
    
    print("✅ RuneLite client found and calibrated!")
    
    # Test multiple screenshots
    for test_num in range(3):
        print(f"\n📸 Test #{test_num + 1}")
        print("-" * 30)
        
        # Capture screenshot of RuneLite client only
        try:
            screenshot = screen_capture.capture_client()
            if screenshot is None:
                print("❌ Failed to capture RuneLite client screenshot")
                continue
                
            print(f"✅ Screenshot captured: {screenshot.shape}")
            
            # Analyze with ULTRA-ADVANCED RuneLite vision
            start_time = time.time()
            results = ultra_advanced_runelite_vision.analyze_runelite_ultra(screenshot)
            analysis_time = time.time() - start_time
            
            print(f"⚡ Analysis completed in {analysis_time:.3f}s")
            
            # Display results
            print_runelite_results(results)
            
            # Save debug image
            debug_image = create_debug_overlay(screenshot, results)
            debug_path = f"debug_runelite_test_{test_num + 1}.png"
            cv2.imwrite(debug_path, debug_image)
            print(f"💾 Debug image saved: {debug_path}")
            
        except Exception as e:
            print(f"❌ Test {test_num + 1} failed: {e}")
            logger.error(f"RuneLite test failed: {e}")
        
        # Wait between tests
        if test_num < 2:
            time.sleep(2)

def print_runelite_results(results: dict):
    """Print formatted RuneLite analysis results"""
    
    # Active tab
    active_tab = results.get('active_tab', RuneLiteTab.UNKNOWN)
    print(f"🎯 Active Tab: {active_tab.value.title()}")
    
    # Orbs status
    orbs = results.get('orbs', {})
    if orbs:
        print("🔮 Orbs Detected:")
        for orb_name, orb_data in orbs.items():
            state = orb_data.get('state', 'unknown')
            confidence = orb_data.get('confidence', 0)
            print(f"   • {orb_name.title()}: {state} ({confidence:.1%})")
    else:
        print("🔮 No orbs detected")
    
    # Inventory items
    items = results.get('inventory_items', [])
    if items:
        print(f"🎒 Inventory: {len(items)} items detected")
        for item in items[:5]:  # Show first 5 items
            slot = item.get('slot', '?')
            confidence = item.get('confidence', 0)
            print(f"   • Slot {slot}: {confidence:.1%} confidence")
    elif active_tab == RuneLiteTab.INVENTORY:
        print("🎒 Inventory tab active but no items detected")
    
    # Prayers
    prayers = results.get('prayers', [])
    if prayers:
        active_prayers = [p for p in prayers if p.get('active', False)]
        print(f"🙏 Prayers: {len(active_prayers)} active, {len(prayers)} total")
    elif active_tab == RuneLiteTab.PRAYER:
        print("🙏 Prayer tab active but no prayers detected")
    
    # Equipment
    equipment = results.get('equipment', {})
    if equipment:
        has_equipment = equipment.get('has_equipment', False)
        confidence = equipment.get('confidence', 0)
        print(f"⚔️ Equipment: {'Equipped' if has_equipment else 'Empty'} ({confidence:.1%})")
    elif active_tab == RuneLiteTab.EQUIPMENT:
        print("⚔️ Equipment tab active but no equipment detected")
    
    # Chat activity
    chat = results.get('chat_messages', [])
    if chat:
        print(f"💬 Chat: {len(chat)} text areas detected")
    else:
        print("💬 No chat activity detected")
    
    # Performance
    performance = results.get('performance', {})
    if performance:
        proc_time = performance.get('processing_time', 0)
        avg_time = performance.get('avg_processing_time', 0)
        screenshot_size = performance.get('screenshot_size', 'unknown')
        print(f"📊 Performance: {proc_time:.3f}s (avg: {avg_time:.3f}s, {screenshot_size})")

def create_debug_overlay(screenshot: np.ndarray, results: dict) -> np.ndarray:
    """Create debug overlay showing detection results"""
    debug_img = screenshot.copy()
    h, w = debug_img.shape[:2]
    
    # Draw active tab indicator
    active_tab = results.get('active_tab', RuneLiteTab.UNKNOWN)
    cv2.putText(debug_img, f"Active Tab: {active_tab.value.title()}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Draw CORRECTED orb regions (in actual OSRS interface area)
    orb_positions = [
        (0.88, 0.05, 0.05, 0.08),  # Health - CORRECTED position
        (0.88, 0.15, 0.05, 0.08),  # Prayer - CORRECTED position
        (0.88, 0.25, 0.05, 0.08),  # Energy - CORRECTED position
        (0.88, 0.35, 0.05, 0.08),  # Special - CORRECTED position
    ]
    
    orb_names = ['Health', 'Prayer', 'Energy', 'Special']
    orbs = results.get('orbs', {})
    
    for i, (x, y, w_ratio, h_ratio) in enumerate(orb_positions):
        start_x = int(w * x)
        start_y = int(h * y)
        end_x = int(w * (x + w_ratio))
        end_y = int(h * (y + h_ratio))
        
        # Draw orb region
        cv2.rectangle(debug_img, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        
        # Show orb state if detected
        orb_key = orb_names[i].lower()
        if orb_key in orbs:
            state = orbs[orb_key].get('state', 'unknown')
            cv2.putText(debug_img, f"{orb_names[i]}: {state}", 
                        (start_x, start_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Draw CORRECTED game viewport boundary (95% instead of 85%)
    game_x = int(w * 0.0)
    game_y = int(h * 0.0)
    game_w = int(w * 0.95)  # CORRECTED: 95% to include full minimap and tabs
    game_h = int(h * 1.0)
    cv2.rectangle(debug_img, (game_x, game_y), (game_x + game_w, game_y + game_h), (255, 0, 255), 3)
    cv2.putText(debug_img, "CORRECTED Game Viewport (95%)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # Draw IN-GAME tab region (bottom strip of interface panel)
    tab_x = int(w * 0.55)
    tab_y = int(h * 0.85)
    tab_w = int(w * 0.25)
    tab_h = int(h * 0.10)
    cv2.rectangle(debug_img, (tab_x, tab_y), (tab_x + tab_w, tab_y + tab_h), (0, 255, 255), 2)
    cv2.putText(debug_img, "In-Game Tabs", (tab_x, tab_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    # Draw individual tab positions for debugging
    tab_positions = [
        (0.565, 0.90, "Combat"),
        (0.595, 0.90, "Stats"), 
        (0.625, 0.90, "Quest"),
        (0.655, 0.90, "Inventory"),
        (0.685, 0.90, "Equipment"),
        (0.715, 0.90, "Prayer"),
        (0.745, 0.90, "Magic"),
        (0.775, 0.90, "Friends"),
    ]
    
    for x_ratio, y_ratio, tab_name in tab_positions:
        tab_center_x = int(w * x_ratio)
        tab_center_y = int(h * y_ratio)
        
        # Draw small circle at each tab position
        cv2.circle(debug_img, (tab_center_x, tab_center_y), 8, (255, 255, 0), 2)
        
        # Show tab name if it's the active tab
        if active_tab.value.lower() == tab_name.lower():
            cv2.circle(debug_img, (tab_center_x, tab_center_y), 12, (0, 255, 0), 3)
            cv2.putText(debug_img, f"ACTIVE: {tab_name}", 
                        (tab_center_x - 30, tab_center_y - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    
    # Draw CORRECTED inventory region if inventory tab is active
    if active_tab == RuneLiteTab.INVENTORY:
        inv_x = int(w * 0.82)  # CORRECTED: Further right to include full interface
        inv_y = int(h * 0.35)
        inv_w = int(w * 0.13)  # CORRECTED: Smaller width since moved further right
        inv_h = int(h * 0.30)
        cv2.rectangle(debug_img, (inv_x, inv_y), (inv_x + inv_w, inv_y + inv_h), (0, 255, 0), 2)
        
        items = results.get('inventory_items', [])
        cv2.putText(debug_img, f"CORRECTED Inventory: {len(items)} items", 
                    (inv_x, inv_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw CORRECTED chat region (full corrected viewport)
    chat_x = int(w * 0.01)
    chat_y = int(h * 0.75)
    chat_w = int(w * 0.94)  # CORRECTED: Extends to full corrected viewport
    chat_h = int(h * 0.24)
    cv2.rectangle(debug_img, (chat_x, chat_y), (chat_x + chat_w, chat_y + chat_h), (255, 255, 0), 2)
    
    chat_messages = results.get('chat_messages', [])
    cv2.putText(debug_img, f"CORRECTED Chat: {len(chat_messages)} areas", 
                (chat_x, chat_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    return debug_img

def test_tab_switching():
    """Test tab detection accuracy by monitoring tab changes"""
    print("\n🔄 Testing Tab Switch Detection")
    print("=" * 40)
    print("Switch between different tabs in RuneLite and watch detection...")
    
    screen_capture = ScreenCapture()
    
    # Make sure client is still calibrated
    if screen_capture.client_region is None:
        print("🎯 Re-calibrating RuneLite client...")
        if not screen_capture.calibrate_client():
            print("❌ Failed to find RuneLite client for tab switching test!")
            return
    
    last_tab = RuneLiteTab.UNKNOWN
    
    for i in range(20):  # Monitor for 20 seconds
        try:
            screenshot = screen_capture.capture_client()
            if screenshot is not None:
                results = ultra_advanced_runelite_vision.analyze_runelite_ultra(screenshot)
                current_tab = results.get('active_tab', RuneLiteTab.UNKNOWN)
                
                if current_tab != last_tab:
                    print(f"🎯 Tab changed: {last_tab.value} → {current_tab.value}")
                    last_tab = current_tab
                    
                    # Show relevant content for new tab
                    if current_tab == RuneLiteTab.INVENTORY:
                        items = results.get('inventory_items', [])
                        print(f"   📦 Detected {len(items)} inventory items")
                    elif current_tab == RuneLiteTab.PRAYER:
                        prayers = results.get('prayers', [])
                        active = [p for p in prayers if p.get('active', False)]
                        print(f"   🙏 Detected {len(active)} active prayers")
                    elif current_tab == RuneLiteTab.EQUIPMENT:
                        equipment = results.get('equipment', {})
                        has_eq = equipment.get('has_equipment', False)
                        print(f"   ⚔️ Equipment detected: {has_eq}")
        
        except Exception as e:
            print(f"❌ Tab monitoring error: {e}")
        
        time.sleep(1)

if __name__ == "__main__":
    print("🚀🧠 ADVANCED RuneLite Vision System Test Suite")
    print("===============================================")
    print("🔥 Cutting-edge ML + Computer Vision + Template Matching")
    print("Make sure RuneLite is open and visible!")
    print()
    
    try:
        # Test basic detection
        test_runelite_detection()
        
        # Test tab switching (interactive)
        input("\nPress Enter to start tab switching test (or Ctrl+C to exit)...")
        test_tab_switching()
        
    except KeyboardInterrupt:
        print("\n\n✅ Testing completed by user")
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        logger.error(f"RuneLite vision testing failed: {e}")
    
    print("\n🎉 RuneLite Vision Testing Complete!") 