# OSRS Bot Framework 🎮

## Overview
A comprehensive, GPU-accelerated framework for creating intelligent Old School RuneScape bots with advanced computer vision, text intelligence, and modern GUI interface.

### Key Features
- 🧠 **AI Text Intelligence** - Real-time OCR and text analysis system
- 👁️ **Advanced Computer Vision** - RuneLite client integration with ML-based detection
- 🖥️ **Modern GUI** - Dark theme CustomTkinter interface with live monitoring
- ⚡ **GPU Acceleration** - RTX 4090 optimized for <100ms processing times
- 🎯 **OSRS-Specific** - Designed specifically for RuneLite and vanilla OSRS clients
- 🛡️ **Safety First** - Human-like behavior patterns and anti-detection measures

## Quick Start

### 1. Installation

**For RTX 4090/GPU Users:**
```bash
git clone https://github.com/your-repo/osrs-all.git
cd osrs-all
pip install -r requirements-gpu.txt
```

**For CPU-Only Users:**
```bash
pip install -r requirements.txt
```

### 2. Launch GUI
```bash
python launch_gui.py
```

### 3. Test Vision System
```bash
python test_runelite_vision.py
```

## Current Status

### ✅ Working Features
- **Text Intelligence System** - Fully functional with GPU acceleration
- **GUI Framework** - Professional interface with real-time monitoring  
- **Screen Capture** - High-performance capture with RuneLite integration
- **OCR Processing** - Chat, items, XP tracking, and interface detection

### 🚧 In Progress
- **Computer Vision Accuracy** - Orb and tab detection needs refinement
- **YOLO Integration** - Transitioning from template matching to ML-based detection
- **AI Decision Systems** - Planning and strategy implementation

## Project Structure

```
osrs-all/
├── core/                    # Framework foundation
│   ├── text_intelligence.py # AI text analysis (✅ Working)
│   ├── screen_capture.py    # High-performance capture
│   └── automation.py        # Human-like controls
├── vision/                  # Computer vision modules
│   ├── osrs_ocr.py         # Text recognition (✅ Working)
│   └── ultra_advanced_vision.py # Latest CV system (needs fixes)
├── gui/                     # Modern interface (✅ Working)
├── bots/                    # Bot implementations
└── tests/                   # Comprehensive test suite
```

## Key Components

### Text Intelligence System
Real-time OCR and analysis with:
- Chat message parsing and filtering
- XP tracking and rate calculations
- Item recognition and value assessment
- Interface state detection
- Alert system for important events

### Computer Vision
Advanced detection capabilities:
- RuneLite client integration
- Interface element detection (tabs, orbs, inventory)
- Multi-resolution support
- Template matching with ML enhancement
- Real-time processing optimization

### GUI Framework
Professional interface featuring:
- Dark theme CustomTkinter design
- Live text intelligence monitoring
- Performance metrics and logging
- Configuration management
- Real-time data visualization

## Usage Examples

### Basic Text Intelligence
```python
from vision.osrs_ocr import osrs_text_intelligence
from core.text_intelligence import text_intelligence

# Capture and analyze
screenshot = capture_osrs_screen()
text_data = osrs_text_intelligence.analyze_game_text(screenshot)
intelligence = text_intelligence.analyze_text_intelligence(text_data)

# Access results
print(f"XP gained: {intelligence['xp_analysis']['session_xp']}")
print(f"Chat messages: {len(text_data['chat_messages'])}")
```

### Computer Vision Detection
```python
from vision.ultra_advanced_vision import UltraAdvancedVision

vision = UltraAdvancedVision()
screenshot = capture_osrs_screen()
game_state = vision.analyze_game_state(screenshot)

print(f"Active tab: {game_state.active_tab}")
print(f"Health: {game_state.player_status.health}")
print(f"Items found: {len(game_state.items)}")
```

## Development Guidelines

### Performance Standards
- Text processing: <100ms
- Computer vision: <50ms per frame
- Memory usage: <2GB
- GPU utilization: >70% (when available)

### Code Quality
- All features must have GUI integration
- Comprehensive test coverage required
- OSRS-specific knowledge and terminology
- Human-like behavior patterns mandatory

### Testing
```bash
# Run all tests
pytest tests/ -v

# Test specific components
pytest tests/test_osrs_ocr.py -v
pytest tests/test_computer_vision.py -v

# Performance benchmarks
pytest tests/ -m performance -v
```

## Architecture & Vision

This framework is designed as the foundation for building the ultimate autonomous OSRS AI agent. The current implementation provides:

1. **Solid Foundation** - Production-ready text intelligence and GUI systems
2. **Extensible Architecture** - Modular design supporting advanced AI integration
3. **Performance Optimization** - GPU acceleration and real-time processing
4. **OSRS Expertise** - Deep understanding of game mechanics and client behavior

### Future Development Phases
- **Phase 2**: Enhanced computer vision with YOLO object detection
- **Phase 3**: AI decision intelligence with goal-oriented planning
- **Phase 4**: Advanced content handling (raids, PvP, questing)
- **Phase 5**: Meta-learning and autonomous progression

## GPU Acceleration

### RTX 4090 Optimization
The framework is specifically optimized for NVIDIA RTX 4090:
- CUDA-enabled PyTorch for text processing
- EasyOCR with GPU acceleration
- Batch processing for multiple detection regions
- Memory-efficient caching systems

### Verification
```bash
# Check GPU availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Monitor performance
python -c "from core.text_intelligence import text_intelligence; print(text_intelligence.get_performance_stats())"
```

## Safety & Ethics

This framework is designed for:
- Educational purposes and computer vision research
- Understanding game automation techniques
- Learning AI and machine learning concepts

**Important**: Always respect game terms of service and play responsibly.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement with comprehensive tests
4. Ensure GUI integration
5. Submit pull request with detailed description

## Dependencies

### Core Requirements
- Python 3.8+
- OpenCV 4.8+
- CustomTkinter
- EasyOCR
- PyTorch (GPU version recommended)

### Optional Enhancements
- scikit-learn (for ML features)
- YOLO (future object detection)
- NumPy/Pandas (data analysis)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Ready to build the future of autonomous gaming?** 🚀

For detailed development guidelines and project overview, see the Cursor Rules in `.cursor/rules/`. 