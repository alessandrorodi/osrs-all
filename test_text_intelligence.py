#!/usr/bin/env python3
"""
Test script for OSRS Text Intelligence
"""

print("Testing OSRS Text Intelligence...")

# 1. Test screen capture
print("\n1. Testing screen capture...")
from core.screen_capture import screen_capture

# First calibrate the client
print("   Calibrating OSRS client...")
calibration_success = screen_capture.calibrate_client()
print(f"   Calibration result: {calibration_success}")

screenshot = screen_capture.capture_client()
if screenshot is not None:
    print(f"✅ Screenshot captured: {screenshot.shape}")
    
    # 2. Test text intelligence
    print("\n2. Testing text intelligence...")
    from vision.osrs_ocr import osrs_text_intelligence
    
    results = osrs_text_intelligence.analyze_game_text(screenshot)
    print("✅ Text analysis completed!")
    
    # 3. Show results
    print("\n3. Results:")
    chat_messages = results.get('chat_messages', [])
    items = results.get('items', [])
    player_stats = results.get('player_stats')
    
    print(f"  📝 Chat messages found: {len(chat_messages)}")
    print(f"  🎒 Items detected: {len(items)}")
    print(f"  👤 Player stats detected: {player_stats is not None}")
    
    if chat_messages:
        print("\n   Recent chat messages:")
        for msg in chat_messages[-5:]:  # Last 5 messages
            print(f"    - {msg.player_name}: {msg.message}")
    
    if items:
        print("\n   Detected items:")
        for item in items[:5]:  # First 5 items
            print(f"    - {item.name} x{item.quantity}")
    
    # 4. Test text intelligence core
    print("\n4. Testing text intelligence analysis...")
    from core.text_intelligence import text_intelligence
    
    analysis = text_intelligence.analyze_text_intelligence(results)
    print("✅ Intelligence analysis completed!")
    
    xp_events = analysis.get('xp_analysis', {}).get('events', [])
    recommendations = analysis.get('recommendations', [])
    alerts = analysis.get('alerts', [])
    
    print(f"  ⚡ XP events: {len(xp_events)}")
    print(f"  💡 Recommendations: {len(recommendations)}")
    print(f"  🚨 Alerts: {len(alerts)}")

else:
    print("❌ Failed to capture screenshot")
    print("Make sure OSRS client is open and calibrated")

print("\nTest completed!") 