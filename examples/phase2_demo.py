#!/usr/bin/env python3
"""
Phase 2 AI Vision System Demonstration

This script demonstrates the capabilities of the Phase 2 AI Vision system
including YOLOv8 object detection, OCR text recognition, and comprehensive
game state analysis.
"""

import sys
import os
import time
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main demonstration function"""
    print("🤖 OSRS Bot Framework - Phase 2 AI Vision Demo")
    print("=" * 60)
    
    try:
        # Import AI Vision components
        print("📦 Importing AI Vision components...")
        from vision.intelligent_vision import intelligent_vision, SceneType
        from core.screen_capture import screen_capture
        
        print("✅ AI Vision system loaded successfully!")
        print(f"   Device: {intelligent_vision.device}")
        print(f"   Game regions configured: {len(intelligent_vision.game_regions)}")
        
        # Check dependencies
        print("\n🔍 Checking dependencies...")
        dependencies = check_dependencies()
        for dep, status in dependencies.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {dep}")
        
        if not all(dependencies.values()):
            print("\n⚠️  Some dependencies are missing. Install them with:")
            print("   pip install ultralytics easyocr torch torchvision")
            print("   Note: The system will fall back to basic computer vision.")
        
        # Demonstrate screen capture
        print("\n📷 Testing screen capture...")
        image = screen_capture.capture_screen()
        if image is not None:
            print(f"✅ Screen captured: {image.shape[1]}x{image.shape[0]} pixels")
            
            # Demonstrate AI Vision analysis
            print("\n🧠 Running AI Vision analysis...")
            start_time = time.time()
            
            game_state = intelligent_vision.analyze_game_state(image)
            
            analysis_time = time.time() - start_time
            print(f"✅ Analysis completed in {analysis_time:.3f} seconds")
            
            # Display results
            print_analysis_results(game_state)
            
            # Performance statistics
            print("\n📊 Performance Statistics:")
            stats = intelligent_vision.get_performance_stats()
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.4f}")
                else:
                    print(f"   {key}: {value}")
            
        else:
            print("❌ Failed to capture screen")
            print("   Make sure you have a display and appropriate permissions")
        
        # Demonstrate individual components
        print("\n🔧 Testing individual components...")
        test_individual_components()
        
    except ImportError as e:
        print(f"❌ Failed to import AI Vision components: {e}")
        print("   Make sure you have installed the required dependencies:")
        print("   pip install ultralytics easyocr torch torchvision")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        logger.exception("Demo error")
    
    print("\n🎯 Phase 2 AI Vision demonstration complete!")
    print("   Check the GUI for interactive features: python launch_gui.py")


def check_dependencies():
    """Check if required dependencies are available"""
    dependencies = {}
    
    try:
        import torch
        dependencies['PyTorch'] = True
    except ImportError:
        dependencies['PyTorch'] = False
    
    try:
        import ultralytics
        dependencies['YOLOv8 (Ultralytics)'] = True
    except ImportError:
        dependencies['YOLOv8 (Ultralytics)'] = False
    
    try:
        import easyocr
        dependencies['EasyOCR'] = True
    except ImportError:
        dependencies['EasyOCR'] = False
    
    try:
        import cv2
        dependencies['OpenCV'] = True
    except ImportError:
        dependencies['OpenCV'] = False
    
    try:
        import numpy
        dependencies['NumPy'] = True
    except ImportError:
        dependencies['NumPy'] = False
    
    return dependencies


def print_analysis_results(game_state):
    """Print comprehensive analysis results"""
    print(f"\n🎯 Game State Analysis Results:")
    print(f"   📅 Timestamp: {time.strftime('%H:%M:%S', time.localtime(game_state.timestamp))}")
    print(f"   🎭 Scene Type: {game_state.scene_type.value}")
    print(f"   🎯 Confidence: {game_state.confidence:.3f}")
    print(f"   ⚡ Processing Time: {game_state.processing_time:.3f}s")
    print(f"   📈 Analysis Version: {game_state.analysis_version}")
    
    print(f"\n👤 Player Status:")
    print(f"   💚 Health: {game_state.player_status.health_percent:.1f}%")
    print(f"   🙏 Prayer: {game_state.player_status.prayer_percent:.1f}%")
    print(f"   ⚡ Energy: {game_state.player_status.energy_percent:.1f}%")
    print(f"   ⚔️  In Combat: {game_state.player_status.is_in_combat}")
    print(f"   🏃 Moving: {game_state.player_status.is_moving}")
    
    print(f"\n🗺️  Minimap Analysis:")
    print(f"   📍 Player Position: {game_state.minimap.player_position}")
    print(f"   🧭 North Direction: {game_state.minimap.north_direction:.1f}°")
    print(f"   🏰 Region Type: {game_state.minimap.region_type}")
    print(f"   👾 NPCs Visible: {len(game_state.minimap.visible_npcs or [])}")
    print(f"   👥 Players Visible: {len(game_state.minimap.visible_players or [])}")
    
    print(f"\n🎒 Inventory Analysis:")
    print(f"   📦 Total Items: {len(game_state.inventory.items or [])}")
    print(f"   📭 Free Slots: {game_state.inventory.free_slots}/28")
    print(f"   💎 Valuable Items: {len(game_state.inventory.valuable_items or [])}")
    print(f"   🍎 Consumables: {len(game_state.inventory.consumables or [])}")
    
    print(f"\n💬 Interface State:")
    print(f"   🖥️  Open Interfaces: {len(game_state.interface_state.open_interfaces or [])}")
    print(f"   💬 Chat Messages: {len(game_state.interface_state.active_chat or [])}")
    print(f"   🖱️  Clickable Elements: {len(game_state.interface_state.clickable_elements or [])}")
    
    print(f"\n🎯 Object Detections:")
    print(f"   👾 NPCs: {len(game_state.npcs or [])}")
    print(f"   💎 Items: {len(game_state.items or [])}")
    print(f"   👥 Players: {len(game_state.players or [])}")
    print(f"   🖥️  UI Elements: {len(game_state.ui_elements or [])}")
    print(f"   🌳 Environment: {len(game_state.environment or [])}")
    
    # Show high priority objects if any
    if hasattr(game_state, 'get_highest_priority_objects'):
        top_objects = game_state.get_highest_priority_objects(3)
        if top_objects:
            print(f"\n🏆 Top Priority Objects:")
            for i, obj in enumerate(top_objects, 1):
                print(f"   {i}. {obj.label} (Priority: {obj.action_priority:.3f}, Type: {obj.object_type})")


def test_individual_components():
    """Test individual AI Vision components"""
    print("🔍 YOLO Detector:")
    try:
        from vision.detectors.yolo_detector import YOLODetector
        detector = YOLODetector(device="cpu")
        print(f"   ✅ Initialized with device: {detector.device}")
        print(f"   🎯 Confidence threshold: {detector.confidence_threshold}")
        print(f"   🎭 Object types: {len(detector.game_object_types)} categories")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
    
    print("\n📝 OCR Detector:")
    try:
        from vision.detectors.ocr_detector import OCRDetector
        detector = OCRDetector(use_gpu=False)
        print(f"   ✅ Initialized with languages: {detector.languages}")
        print(f"   🎯 Text patterns: {len(detector.osrs_patterns)} types")
        print(f"   🔧 Preprocessing methods: {len(detector.preprocessing_methods)}")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
    
    print("\n🗺️  Minimap Analyzer:")
    try:
        from vision.intelligent_vision import MinimapAnalyzer
        analyzer = MinimapAnalyzer()
        print(f"   ✅ Initialized with region: {analyzer.minimap_region}")
        print(f"   🎨 Dot colors: {len(analyzer.dot_colors)} types")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")
    
    print("\n🎭 Scene Classifier:")
    try:
        from vision.intelligent_vision import SceneClassifier
        classifier = SceneClassifier()
        print(f"   ✅ Initialized with {len(classifier.scene_indicators)} scene types")
        for scene_type in classifier.scene_indicators.keys():
            print(f"      - {scene_type.value}")
    except Exception as e:
        print(f"   ❌ Failed to initialize: {e}")


if __name__ == "__main__":
    main()