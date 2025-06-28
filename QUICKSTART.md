# OSRS Bot Framework - Quick Start Guide

This guide will help you get started with the OSRS Bot Framework quickly.

## Prerequisites

1. **Python 3.8+** installed on your system
2. **Old School RuneScape** client (recommended: official client)
3. **Windows 10/11** (framework designed for Windows, but may work on other OS)

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run initial setup:**
   ```bash
   python setup.py
   ```

   This will:
   - Create necessary directories
   - Test all dependencies
   - Verify screen capture functionality
   - Create example configuration files

## Initial Configuration

### Step 1: Calibrate Your Client

1. **Open OSRS client** and position it where you want it during botting
2. **Run calibration tool:**
   ```bash
   python tools/calibrate_client.py
   ```
3. Follow the prompts to calibrate screen capture

### Step 2: Create Templates

Templates are image snippets used for object detection. You'll need to create templates for:
- Monsters you want to target
- Food items
- UI elements
- Any other objects your bot needs to recognize

1. **Run template creator:**
   ```bash
   python tools/template_creator.py
   ```

2. **Create essential templates:**
   - Select option 1 to create new templates
   - Capture screenshots of your OSRS client
   - Select objects by clicking and dragging
   - Save with descriptive names like:
     - `goblin` - for goblin monsters
     - `bread` - for bread food
     - `health_bar` - for health indicators

3. **Test your templates:**
   - Select option 3 to test template matching
   - Verify that your templates are detected correctly

## Running Your First Bot

### Option 1: Run the Example Bot

1. **Create required templates first:**
   - `goblin` (or change the target in the bot)
   - `bread` (or change the food item)
   - `health_bar`

2. **Run the example bot:**
   ```bash
   python bots/example_bot.py
   ```

3. **Monitor the bot:**
   - Watch console output for actions and status
   - Press `Ctrl+C` to stop the bot safely

### Option 2: Create Your Own Bot

1. **Create a new bot file** (e.g., `bots/my_bot.py`)

2. **Basic bot structure:**
   ```python
   from core.bot_base import BotBase
   from utils.logging import BotLogger

   class MyBot(BotBase):
       def __init__(self, config=None):
           super().__init__("MyBot", config)
           self.logger = BotLogger("my_bot")
       
       def initialize(self) -> bool:
           """Setup your bot here"""
           self.logger.log_action("Bot initializing")
           return True
       
       def execute_cycle(self) -> bool:
           """Main bot logic goes here"""
           # Your bot logic
           return True
   
   if __name__ == "__main__":
       bot = MyBot()
       bot.start()
   ```

## Framework Features

### Computer Vision
- **Template matching** - Find UI elements and objects
- **Color detection** - Detect based on color (health bars, etc.)
- **Feature detection** - Advanced object recognition

### Automation
- **Human-like mouse movements** - Bezier curves and randomization
- **Smart clicking** - Confidence-based targeting
- **Keyboard automation** - Text input and key combinations

### Safety Features
- **Emergency stops** - Multiple failsafe mechanisms
- **Anti-detection** - Randomized timing and breaks
- **Focus monitoring** - Ensure client is active
- **Error handling** - Comprehensive error recovery

### Logging and Monitoring
- **Detailed logging** - Actions, errors, and performance
- **Performance tracking** - Actions/minute, success rates
- **Visual debugging** - Screenshot and overlay tools

## Configuration

### Bot Configuration
Edit `config/example_bot.yaml` or create your own:

```yaml
name: "my_bot"
enabled: true

settings:
  target_monster: "goblin"
  food_item: "bread"
  health_threshold: 50

templates:
  - "goblin"
  - "bread"
  - "health_bar"

regions:
  inventory: [548, 205, 190, 261]
  minimap: [570, 9, 146, 151]
  chat: [7, 345, 506, 120]
```

### Framework Settings
Modify `config/settings.py` for:
- Screen capture settings
- Computer vision thresholds
- Automation timing
- Safety parameters

## Troubleshooting

### Common Issues

1. **"OSRS client not found"**
   - Ensure OSRS client is open
   - Check window title matches "Old School RuneScape"
   - Try running as administrator

2. **"Template not found"**
   - Create templates using `tools/template_creator.py`
   - Ensure template names match your bot configuration
   - Test templates with option 3 in template creator

3. **"Screen capture failed"**
   - Check if other applications are blocking screen capture
   - Try running as administrator
   - Ensure client is visible and not minimized

4. **Bot stops immediately**
   - Check console output for error messages
   - Verify client is focused (if require_focus is True)
   - Ensure all required templates exist

### Getting Help

1. **Check logs:** Look in the `logs/` directory for detailed error information
2. **Debug mode:** Enable debug settings in `config/settings.py`
3. **Visual debugging:** Enable visual overlays to see what the bot detects

## Next Steps

1. **Study the example bot** to understand the framework
2. **Create templates** for your specific use case
3. **Build your own bot** by extending `BotBase`
4. **Test thoroughly** in safe environments
5. **Add safety measures** appropriate for your bot

## Safety and Ethics

- **Use responsibly** - This is for educational purposes
- **Respect ToS** - Follow Jagex's terms of service
- **Test safely** - Use accounts you don't mind losing
- **Consider others** - Don't negatively impact other players

## Advanced Features

Once comfortable with basics, explore:
- **Custom detectors** in `vision/detectors/`
- **Machine learning** models for advanced recognition
- **Multi-bot coordination** 
- **Performance optimization**
- **Advanced computer vision** techniques

---

**Ready to start botting? Run `python setup.py` and follow this guide!** 