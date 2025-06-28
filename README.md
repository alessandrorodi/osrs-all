# OSRS Bot Framework

A production-ready, modular framework for creating Old School RuneScape automation bots using advanced computer vision, machine learning, and human-like automation.

## 🎯 **Features**

### **Core Framework**
- **🤖 Modular Architecture**: Easy to create new bots by extending base classes
- **👁️ Advanced Computer Vision**: Template matching, object detection, and visual debugging
- **🖱️ Human-like Automation**: Bezier curve movements, realistic timing, anti-detection
- **🛡️ Safety Systems**: Emergency stops, failsafes, and behavior randomization
- **📊 Performance Monitoring**: Real-time analytics, success rates, and optimization
- **🎮 OSRS Specific**: Optimized for Old School RuneScape client detection

### **Modern GUI Interface**
- **🖥️ Dark Theme**: Professional CustomTkinter interface
- **📱 Real-time Dashboard**: Live system monitoring and bot management
- **📊 Performance Charts**: Visual analytics and performance tracking
- **👁️ Computer Vision Testing**: Live detection preview and debugging
- **📋 Log Management**: Integrated log viewer with filtering
- **🖼️ Template Browser**: Visual template management and creation

## 🚀 **Quick Start**

### **Installation**
```bash
# Clone the repository
git clone <your-repo-url>
cd osrs-bot-framework

# Install dependencies
pip install -r requirements.txt

# Run initial setup
python setup.py
```

### **Launch GUI (Recommended)**
```bash
python launch_gui.py
```

### **Command Line Setup**
```bash
# 1. Calibrate OSRS client
python tools/calibrate_client.py

# 2. Create detection templates
python tools/template_creator.py

# 3. Run example bot
python bots/example_bot.py
```

## 🖥️ **GUI Interface**

### **Dashboard Tab**
- **🔴/🟢 System Status**: OSRS client connection, screen capture, computer vision
- **📊 Quick Stats**: Performance metrics, bot count, system health
- **🛑 Emergency Controls**: Immediate stop for all running bots

### **Bots Tab**
- **➕ Bot Management**: Create, load, and configure bots
- **📊 Bot List**: Status, runtime, actions performed, error counts
- **▶️ Controls**: Start, pause, stop individual bots

### **Vision Tab**
- **👁️ Live Testing**: Real-time computer vision with overlay
- **📷 Detection**: Test template matching and object recognition
- **🎯 Debug View**: Confidence scores and bounding boxes

### **Performance Tab**
- **📈 Real-time Charts**: Actions/min, success rates, runtime tracking
- **📊 Analytics**: Multi-bot comparison and performance optimization
- **💾 Data Export**: Save performance metrics for analysis

### **Logs Tab**
- **📋 Live Logs**: Real-time log viewing with syntax highlighting
- **🔍 Filtering**: Filter by level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **💾 Export**: Save logs for debugging and analysis

### **Templates Tab**
- **🖼️ Template Browser**: Visual preview of all detection templates
- **➕ Create Templates**: Launch template creation tool
- **📂 Import/Export**: Manage template files

## 🏗️ **Architecture**

```
osrs-bot-framework/
├── core/              # Framework core components
│   ├── screen_capture.py    # High-performance screen capture
│   ├── automation.py        # Human-like mouse/keyboard control
│   ├── computer_vision.py   # Template matching and detection
│   └── bot_base.py         # Abstract bot base class
├── gui/               # Modern GUI interface
│   ├── main_window.py      # Main application window
│   ├── tabs.py            # Tab management system
│   └── handlers.py        # Event handling
├── vision/            # Computer vision modules
│   └── detectors/         # Specialized detection algorithms
├── bots/              # Bot implementations
│   └── example_bot.py     # Example combat bot
├── tools/             # Development and debugging tools
│   ├── calibrate_client.py # OSRS client calibration
│   └── template_creator.py # Interactive template creation
├── config/            # Configuration files
│   ├── settings.py        # Framework settings
│   └── *.yaml            # Bot configurations
├── utils/             # Utility functions
│   └── logging.py         # Comprehensive logging system
└── data/              # Templates, models, and data
    ├── templates/         # Detection template images
    └── models/           # AI models (future)
```

## 🤖 **Creating Your First Bot**

### **1. Basic Bot Structure**
```python
from core.bot_base import BotBase
from utils.logging import get_logger

class MyBot(BotBase):
    def __init__(self, config=None):
        super().__init__("MyBot", config)
        self.logger = get_logger(__name__)
    
    def initialize(self) -> bool:
        """Setup your bot here"""
        self.logger.info("Bot initializing")
        return True
    
    def execute_cycle(self) -> bool:
        """Main bot logic goes here"""
        # Your bot logic
        if self.should_eat():
            self.eat_food()
        if self.target_available():
            self.attack_target()
        return True
    
    def should_eat(self) -> bool:
        """Check if we need to eat food"""
        # Implement health checking logic
        pass
    
    def eat_food(self):
        """Eat food from inventory"""
        # Implement eating logic
        pass

if __name__ == "__main__":
    bot = MyBot()
    bot.start()
```

### **2. Bot Configuration (YAML)**
```yaml
name: "my_combat_bot"
enabled: true

settings:
  target_monster: "goblin"
  food_item: "bread"
  health_threshold: 50
  combat_style: "aggressive"

templates:
  - "goblin"
  - "bread" 
  - "health_bar"

regions:
  inventory: [548, 205, 190, 261]
  minimap: [570, 9, 146, 151]
  chat: [7, 345, 506, 120]

safety:
  emergency_health: 20
  max_runtime: 3600  # 1 hour
  break_interval: 1800  # 30 minutes
```

## 👁️ **Computer Vision**

### **Template Matching**
```python
from core.computer_vision import ComputerVision

cv = ComputerVision()

# Find objects in screenshot
results = cv.find_template("goblin", screenshot, confidence=0.8)
if results:
    x, y, confidence = results[0]
    print(f"Found goblin at ({x}, {y}) with {confidence:.2f} confidence")
```

### **Creating Templates**
1. **Launch template creator**: `python tools/template_creator.py`
2. **Capture screenshot** of your OSRS client
3. **Select objects** by clicking and dragging
4. **Save with descriptive names** (e.g., "goblin", "bread", "bank_booth")
5. **Test templates** to verify detection accuracy

## 🛡️ **Safety Features**

### **Anti-Detection**
- **Human-like movements**: Bezier curves with random variations
- **Realistic timing**: Random delays and reaction times
- **Behavior patterns**: Breaks, mistakes, and natural variations
- **Focus monitoring**: Ensures client window is active

### **Emergency Controls**
- **Emergency stop**: `Ctrl+C` or GUI emergency button
- **Auto-logout**: On suspicious activity or errors
- **Health monitoring**: Automatic food consumption
- **Runtime limits**: Configurable maximum bot runtime

### **Error Handling**
- **Screenshot on error**: Automatic debug image capture
- **Graceful recovery**: Attempt to recover from errors
- **Comprehensive logging**: Detailed error tracking
- **Failsafe mechanisms**: Multiple layers of protection

## 📊 **Performance Monitoring**

### **Real-time Metrics**
- **Actions per minute**: Bot efficiency tracking
- **Success rate**: Percentage of successful actions
- **Error rate**: Errors per hour tracking
- **Runtime statistics**: Uptime and performance history

### **Visual Analytics**
- **Performance charts**: Real-time graphical monitoring
- **Comparison tools**: Multi-bot performance analysis
- **Export capabilities**: Save data for external analysis
- **Optimization suggestions**: Automated performance tips

## 🔧 **Configuration**

### **Framework Settings** (`config/settings.py`)
```python
# Screen capture settings
SCREEN_CAPTURE = {
    'method': 'mss',  # or 'pyautogui'
    'region': None,   # Auto-detect or specify region
    'fps_limit': 30
}

# Computer vision settings
VISION = {
    'template_threshold': 0.8,
    'debug_images': True,
    'max_detections': 10
}

# Automation settings
AUTOMATION = {
    'mouse_speed': 'human',  # 'fast', 'human', 'slow'
    'click_delay': (0.1, 0.3),
    'movement_variation': 0.1
}
```

### **Bot-Specific Settings**
Each bot can have its own YAML configuration file with:
- Target specifications
- Inventory management
- Combat settings
- Safety parameters
- Performance tuning

## 🚨 **Troubleshooting**

### **Common Issues**

**"OSRS client not found"**
- Ensure OSRS client is open and visible
- Run `python tools/calibrate_client.py` to detect client
- Check window title matches "Old School RuneScape"

**"Template not found"**
- Create templates using `python tools/template_creator.py`
- Verify template names match bot configuration
- Test templates with Vision tab in GUI

**"Permission denied" errors**
- Run as administrator (Windows)
- Check antivirus software blocking screen capture
- Ensure Python has necessary permissions

**Poor detection accuracy**
- Create multiple templates for same object
- Adjust confidence thresholds in settings
- Test with different graphics settings
- Use Vision tab to debug detection

### **Getting Help**
1. **Check logs**: Look in `logs/` directory for detailed error information
2. **Enable debug mode**: Set `debug_images: True` in settings
3. **Use GUI debugging**: Vision tab shows live detection results
4. **Test components**: Use individual tools to isolate issues

## 🌟 **Advanced Features**

### **Multi-Bot Management**
- Run multiple bots simultaneously
- Coordinated bot strategies
- Resource sharing and conflict resolution
- Performance monitoring across all bots

### **Custom Detectors**
- Extend computer vision capabilities
- Create specialized detection algorithms
- Integrate machine learning models
- Add new object recognition types

### **Plugin System**
- Modular bot components
- Shared functionality libraries
- Community-developed extensions
- Easy integration of new features

## 🛣️ **Roadmap**

See [ROADMAP.md](ROADMAP.md) for the complete vision of evolving this framework into the ultimate AI gaming agent, including:

- **Phase 2**: YOLOv8 object detection, OCR text recognition
- **Phase 3**: Advanced AI decision making and learning
- **Phase 4**: Complex content handling (raids, PvP, questing)
- **Phase 5**: Meta-learning and autonomous gameplay

## ⚖️ **Legal & Ethics**

**This framework is for educational and personal use only.**

- **Respect Jagex's Terms of Service**
- **Use at your own risk** - account bans are possible
- **Consider impact on other players** - be respectful
- **Educational purpose** - learn about AI, automation, and computer vision
- **No guarantees** - framework provided as-is

## 📜 **License**

This project is for educational and personal use only. See the full terms in the project license.

---

## 🎉 **Ready to Start?**

1. **Install**: Follow the installation steps above
2. **Launch GUI**: Run `python launch_gui.py`
3. **Calibrate**: Use the calibration tool to detect your OSRS client
4. **Create Templates**: Build your object detection library
5. **Build Bots**: Start with the example bot and create your own
6. **Monitor**: Use the GUI to track performance and debug issues

**Welcome to the future of OSRS automation!** 🚀 